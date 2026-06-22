#import <Metal/Metal.h>
#include <cstring>
#include "../models/sv.hpp"
#include "../models/qmdd.hpp"
#include "backend.hpp"

struct GPUInputHeader {
    uint32_t kind;
    float   root_re;
    float   root_im;
    uint32_t dim;
};

static GPUInputHeader makeHeader(const GPUInput& in) {
    GPUInputHeader h{};
    h.kind = (uint32_t)in.kind;
    h.root_re = (float)in.root_re;
    h.root_im = (float)in.root_im;
    h.dim = (uint32_t)in.dim;
    return h;
}

struct MetalContext {
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> queue;
};

static MetalContext& getMetalContext() {
    static MetalContext ctx = [] {
        MetalContext c{};
        c.device = MTLCreateSystemDefaultDevice();
        NSURL* libURL = [NSURL fileURLWithPath:@"/Users/mitsuishikaito/my_quantum_simulator_with_gpu/build/Heian.metallib"];
        NSError* err = nil;
        c.library = [c.device newLibraryWithURL:libURL error:&err];
        c.queue = [c.device newCommandQueue];
        return c;
    }();
    return ctx;
}

static id<MTLComputePipelineState> getPSO(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* fnName
) {
    static std::mutex mtx;
    static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* cache = nil;

    std::lock_guard<std::mutex> lock(mtx);

    if (!cache) {
        cache = [NSMutableDictionary new];
    }

    id<MTLComputePipelineState> cached = cache[fnName];
    if (cached) return cached;

    id<MTLFunction> fn = [library newFunctionWithName:fnName];
    if (!fn) {
        NSLog(@"getPSO: function not found: %@", fnName);
        return nil;
    }

    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        NSLog(@"getPSO: PSO creation failed for %@: %@", fnName, err);
        return nil;
    }

    cache[fnName] = pso;
    return pso;
}


void runHash2Pass(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> outReBuf,
    id<MTLBuffer> outImBuf,
    id<MTLBuffer> outIdBuf,
    uint32_t total
) {
    id<MTLFunction> p1Fn = [library newFunctionWithName:@"hash_pass1"];
    id<MTLFunction> p2Fn = [library newFunctionWithName:@"hash_pass2"];
    id<MTLComputePipelineState> p1PSO = getPSO(device, library, @"hash_pass1");
    id<MTLComputePipelineState> p2PSO = getPSO(device, library, @"hash_pass2");

    NSUInteger tg = p1PSO.maxTotalThreadsPerThreadgroup;
    NSUInteger groups = (total + tg - 1) / tg;

    id<MTLBuffer> partialBuf =
        [device newBufferWithLength:sizeof(uint64_t)*groups options:MTLResourceStorageModeShared];

    id<MTLBuffer> totalBuf =
        [device newBufferWithBytes:&total length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    id<MTLBuffer> groupBuf =
        [device newBufferWithBytes:&groups length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    // pass1
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p1PSO];
        [enc setBuffer:outReBuf   offset:0 atIndex:0];
        [enc setBuffer:outImBuf   offset:0 atIndex:1];
        [enc setBuffer:partialBuf offset:0 atIndex:2];
        [enc setBuffer:totalBuf   offset:0 atIndex:3];

        MTLSize grid = MTLSizeMake(groups * tg, 1, 1);
        MTLSize tgs  = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
    }

    // pass2
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p2PSO];
        [enc setBuffer:partialBuf offset:0 atIndex:0];
        [enc setBuffer:outIdBuf   offset:0 atIndex:1];
        [enc setBuffer:groupBuf   offset:0 atIndex:2];

        MTLSize grid = MTLSizeMake(1, 1, 1);
        MTLSize tgs  = MTLSizeMake(1, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
    }
}

static void runMulAny2(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    id<MTLCommandQueue> queue,
    const GPUInputHeader& hdrA,
    const GPUInputHeader& hdrB,
    const void* edgesA, size_t edgesA_bytes,
    const void* edgesB, size_t edgesB_bytes,
    id<MTLBuffer> inReBufA,
    id<MTLBuffer> inImBufA,
    id<MTLBuffer> inReBufB,
    id<MTLBuffer> inImBufB,
    id<MTLBuffer> outReBuf,
    id<MTLBuffer> outImBuf,
    id<MTLBuffer> outIdBuf
) {
    // id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError* error = nil;
    id<MTLComputePipelineState> pso = getPSO(device, library, @"mul_any2");

    id<MTLBuffer> hdrBufA = [device newBufferWithBytes:&hdrA length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdrBufB = [device newBufferWithBytes:&hdrB length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];

    id<MTLBuffer> edgesBufA = edgesA_bytes ? [device newBufferWithBytes:edgesA length:edgesA_bytes options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> edgesBufB = edgesB_bytes ? [device newBufferWithBytes:edgesB length:edgesB_bytes options:MTLResourceStorageModeShared] : nil;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];

    [enc setBuffer:hdrBufA   offset:0 atIndex:0];
    [enc setBuffer:hdrBufB   offset:0 atIndex:1];
    [enc setBuffer:edgesBufA offset:0 atIndex:2];
    [enc setBuffer:edgesBufB offset:0 atIndex:3];
    [enc setBuffer:inReBufA  offset:0 atIndex:4];
    [enc setBuffer:inImBufA  offset:0 atIndex:5];
    [enc setBuffer:inReBufB  offset:0 atIndex:6];
    [enc setBuffer:inImBufB  offset:0 atIndex:7];
    [enc setBuffer:outReBuf  offset:0 atIndex:8];
    [enc setBuffer:outImBuf  offset:0 atIndex:9];

    uint32_t dim = hdrA.dim;
    uint32_t total = dim * dim;

    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    MTLSize tgs = MTLSizeMake(tg, 1, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    [enc endEncoding];

    runHash2Pass(device, library, cmd, outReBuf, outImBuf, outIdBuf, total);

    [cmd commit];
    [cmd waitUntilCompleted];
}

static void runAddAny2(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    id<MTLCommandQueue> queue,
    const GPUInputHeader& hdrA,
    const GPUInputHeader& hdrB,
    const void* edgesA, size_t edgesA_bytes,
    const void* edgesB, size_t edgesB_bytes,
    id<MTLBuffer> inReBufA,
    id<MTLBuffer> inImBufA,
    id<MTLBuffer> inReBufB,
    id<MTLBuffer> inImBufB,
    id<MTLBuffer> outReBuf,
    id<MTLBuffer> outImBuf,
    id<MTLBuffer> outIdBuf
) {
    // id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError* error = nil;
    id<MTLComputePipelineState> pso = getPSO(device, library, @"add_any2");

    id<MTLBuffer> hdrBufA = [device newBufferWithBytes:&hdrA length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdrBufB = [device newBufferWithBytes:&hdrB length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];

    id<MTLBuffer> edgesBufA = edgesA_bytes ? [device newBufferWithBytes:edgesA length:edgesA_bytes options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> edgesBufB = edgesB_bytes ? [device newBufferWithBytes:edgesB length:edgesB_bytes options:MTLResourceStorageModeShared] : nil;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];

    [enc setBuffer:hdrBufA   offset:0 atIndex:0];
    [enc setBuffer:hdrBufB   offset:0 atIndex:1];
    [enc setBuffer:edgesBufA offset:0 atIndex:2];
    [enc setBuffer:edgesBufB offset:0 atIndex:3];
    [enc setBuffer:inReBufA  offset:0 atIndex:4];
    [enc setBuffer:inImBufA  offset:0 atIndex:5];
    [enc setBuffer:inReBufB  offset:0 atIndex:6];
    [enc setBuffer:inImBufB  offset:0 atIndex:7];
    [enc setBuffer:outReBuf  offset:0 atIndex:8];
    [enc setBuffer:outImBuf  offset:0 atIndex:9];

    uint32_t dim = hdrA.dim;
    uint32_t total = dim * dim;

    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    MTLSize tgs = MTLSizeMake(tg, 1, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    [enc endEncoding];

    runHash2Pass(device, library, cmd, outReBuf, outImBuf, outIdBuf, total);

    [cmd commit];
    [cmd waitUntilCompleted];

}

static void runKronAny2(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    id<MTLCommandQueue> queue,
    const GPUInputHeader& hdrA,
    const GPUInputHeader& hdrB,
    const void* edgesA, size_t edgesA_bytes,
    const void* edgesB, size_t edgesB_bytes,
    id<MTLBuffer> inReBufA,
    id<MTLBuffer> inImBufA,
    id<MTLBuffer> inReBufB,
    id<MTLBuffer> inImBufB,
    id<MTLBuffer> outReBuf,
    id<MTLBuffer> outImBuf,
    id<MTLBuffer> outIdBuf
) {
    // id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError* error = nil;
    id<MTLComputePipelineState> pso = getPSO(device, library, @"kron_any2");

    id<MTLBuffer> hdrBufA = [device newBufferWithBytes:&hdrA length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdrBufB = [device newBufferWithBytes:&hdrB length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];

    id<MTLBuffer> edgesBufA = edgesA_bytes ? [device newBufferWithBytes:edgesA length:edgesA_bytes options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> edgesBufB = edgesB_bytes ? [device newBufferWithBytes:edgesB length:edgesB_bytes options:MTLResourceStorageModeShared] : nil;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];

    [enc setBuffer:hdrBufA   offset:0 atIndex:0];
    [enc setBuffer:hdrBufB   offset:0 atIndex:1];
    [enc setBuffer:edgesBufA offset:0 atIndex:2];
    [enc setBuffer:edgesBufB offset:0 atIndex:3];
    [enc setBuffer:inReBufA  offset:0 atIndex:4];
    [enc setBuffer:inImBufA  offset:0 atIndex:5];
    [enc setBuffer:inReBufB  offset:0 atIndex:6];
    [enc setBuffer:inImBufB  offset:0 atIndex:7];
    [enc setBuffer:outReBuf  offset:0 atIndex:8];
    [enc setBuffer:outImBuf  offset:0 atIndex:9];

    uint32_t dimOut = hdrA.dim * hdrB.dim;
    uint32_t total = dimOut * dimOut;

    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    MTLSize tgs = MTLSizeMake(tg, 1, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    [enc endEncoding];

    runHash2Pass(device, library, cmd, outReBuf, outImBuf, outIdBuf, total);

    [cmd commit];
    [cmd waitUntilCompleted];
}

extern "C" void runMulAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    void** outRe,
    void** outIm,
    int64_t* outId,
    float* outCoef
) {
    @autoreleasepool {
        auto& ctx = getMetalContext();

        GPUInputHeader hdrA = makeHeader(A);
        GPUInputHeader hdrB = makeHeader(B);

        const void* edgesA = A.qmdd.edges.empty() ? nullptr : A.qmdd.edges.data();
        size_t edgesA_bytes = A.qmdd.edges.size() * sizeof(GPUEdge);

        const void* edgesB = B.qmdd.edges.empty() ? nullptr : B.qmdd.edges.data();
        size_t edgesB_bytes = B.qmdd.edges.size() * sizeof(GPUEdge);

        id<MTLBuffer> inReBufA = (A.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)A.sv.reHandle : nil;
        id<MTLBuffer> inImBufA = (A.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)A.sv.imHandle : nil;
        id<MTLBuffer> inReBufB = (B.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)B.sv.reHandle : nil;
        id<MTLBuffer> inImBufB = (B.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)B.sv.imHandle : nil;

        size_t inA_bytes = (A.kind == SonKind::SVLeaf) ? A.dim * A.dim * sizeof(float) : 0;
        size_t inB_bytes = (B.kind == SonKind::SVLeaf) ? B.dim * B.dim * sizeof(float) : 0;

        uint32_t dim = A.dim;
        size_t total = (size_t)dim * dim;
        size_t outBytes = total * sizeof(float);

        id<MTLBuffer> outReBuf = [ctx.device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> outImBuf = [ctx.device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];

        id<MTLBuffer> outIdBuf = [ctx.device newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];


        runMulAny2(
            ctx.device, ctx.library, ctx.queue,
            hdrA, hdrB,
            edgesA, edgesA_bytes,
            edgesB, edgesB_bytes,
            inReBufA,
            inImBufA,
            inReBufB,
            inImBufB,
            outReBuf,
            outImBuf,
            outIdBuf
        );

        if (outRe) { [outReBuf retain]; *outRe = (__bridge void*)outReBuf; }
        if (outIm) { [outImBuf retain]; *outIm = (__bridge void*)outImBuf; }

        if (outId) {
            *outId = *(int64_t*)outIdBuf.contents;
        }

        if (outCoef) {
            float rA = (float)A.root_re, iA = (float)A.root_im;
            float rB = (float)B.root_re, iB = (float)B.root_im;
            outCoef[0] = rA*rB - iA*iB;
            outCoef[1] = rA*iB + iA*rB;
        }

        float* re = (float*)outReBuf.contents;
        float* im = (float*)outImBuf.contents;
        // NSLog(@"out[0]=(%f,%f) out[1]=(%f,%f) out[2]=(%f,%f)",
        //     re[0], im[0], re[1], im[1], re[2], im[2]);
        // NSLog(@"outCoef = (%f, %f)", outCoef[0], outCoef[1]);
    }
}

extern "C" void runAddAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    void** outRe,
    void** outIm,
    int64_t* outId,
    float* outCoef
) {
    @autoreleasepool {
        auto& ctx = getMetalContext();

        GPUInputHeader hdrA = makeHeader(A);
        GPUInputHeader hdrB = makeHeader(B);

        const void* edgesA = A.qmdd.edges.empty() ? nullptr : A.qmdd.edges.data();
        size_t edgesA_bytes = A.qmdd.edges.size() * sizeof(GPUEdge);

        const void* edgesB = B.qmdd.edges.empty() ? nullptr : B.qmdd.edges.data();
        size_t edgesB_bytes = B.qmdd.edges.size() * sizeof(GPUEdge);

        id<MTLBuffer> inReBufA = (A.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)A.sv.reHandle : nil;
        id<MTLBuffer> inImBufA = (A.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)A.sv.imHandle : nil;
        id<MTLBuffer> inReBufB = (B.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)B.sv.reHandle : nil;
        id<MTLBuffer> inImBufB = (B.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)B.sv.imHandle : nil;

        size_t inA_bytes = (A.kind == SonKind::SVLeaf) ? A.dim * A.dim * sizeof(float) : 0;
        size_t inB_bytes = (B.kind == SonKind::SVLeaf) ? B.dim * B.dim * sizeof(float) : 0;

        uint32_t dim = A.dim;
        size_t total = (size_t)dim * dim;
        size_t outBytes = total * sizeof(float);

        id<MTLBuffer> outReBuf = [ctx.device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> outImBuf = [ctx.device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];

        id<MTLBuffer> outIdBuf = [ctx.device newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];

        runAddAny2(
            ctx.device, ctx.library, ctx.queue,
            hdrA, hdrB,
            edgesA, edgesA_bytes,
            edgesB, edgesB_bytes,
            inReBufA,
            inImBufA,
            inReBufB,
            inImBufB,
            outReBuf,
            outImBuf,
            outIdBuf
        );

        if (outRe) { [outReBuf retain]; *outRe = (__bridge void*)outReBuf; }
        if (outIm) { [outImBuf retain]; *outIm = (__bridge void*)outImBuf; }

        if (outId) {
            *outId = *(int64_t*)outIdBuf.contents;
        }

        if (outCoef) {
            outCoef[0] = 1.0f;
            outCoef[1] = 0.0f;
        }

        float* re = (float*)outReBuf.contents;
        float* im = (float*)outImBuf.contents;
        // NSLog(@"out[0]=(%f,%f) out[1]=(%f,%f) out[2]=(%f,%f)",
        //     re[0], im[0], re[1], im[1], re[2], im[2]);
        // NSLog(@"outCoef = (%f, %f)", outCoef[0], outCoef[1]);
    }
}

extern "C" void runKronAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    void** outRe,
    void** outIm,
    int64_t* outId,
    float* outCoef
) {
    @autoreleasepool {
        auto& ctx = getMetalContext();

        GPUInputHeader hdrA = makeHeader(A);
        GPUInputHeader hdrB = makeHeader(B);

        const void* edgesA = A.qmdd.edges.empty() ? nullptr : A.qmdd.edges.data();
        size_t edgesA_bytes = A.qmdd.edges.size() * sizeof(GPUEdge);

        const void* edgesB = B.qmdd.edges.empty() ? nullptr : B.qmdd.edges.data();
        size_t edgesB_bytes = B.qmdd.edges.size() * sizeof(GPUEdge);

        id<MTLBuffer> inReBufA = (A.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)A.sv.reHandle : nil;
        id<MTLBuffer> inImBufA = (A.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)A.sv.imHandle : nil;
        id<MTLBuffer> inReBufB = (B.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)B.sv.reHandle : nil;
        id<MTLBuffer> inImBufB = (B.kind == SonKind::SVLeaf) ? (__bridge id<MTLBuffer>)B.sv.imHandle : nil;

        size_t inA_bytes = (A.kind == SonKind::SVLeaf) ? A.dim * A.dim * sizeof(float) : 0;
        size_t inB_bytes = (B.kind == SonKind::SVLeaf) ? B.dim * B.dim * sizeof(float) : 0;

        uint32_t dimOut = A.dim * B.dim;
        size_t total = (size_t)dimOut * dimOut;
        size_t outBytes = total * sizeof(float);

        id<MTLBuffer> outReBuf = [ctx.device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> outImBuf = [ctx.device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];

        id<MTLBuffer> outIdBuf = [ctx.device newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];

        runKronAny2(
            ctx.device, ctx.library, ctx.queue,
            hdrA, hdrB,
            edgesA, edgesA_bytes,
            edgesB, edgesB_bytes,
            inReBufA,
            inImBufA,
            inReBufB,
            inImBufB,
            outReBuf,
            outImBuf,
            outIdBuf
        );

        if (outRe) { [outReBuf retain]; *outRe = (__bridge void*)outReBuf; }
        if (outIm) { [outImBuf retain]; *outIm = (__bridge void*)outImBuf; }

        if (outId) {
            *outId = *(int64_t*)outIdBuf.contents;
        }

        if (outCoef) {
            float rA = (float)A.root_re, iA = (float)A.root_im;
            float rB = (float)B.root_re, iB = (float)B.root_im;
            outCoef[0] = rA*rB - iA*iB;
            outCoef[1] = rA*iB + iA*rB;
        }

        float* re = (float*)outReBuf.contents;
        float* im = (float*)outImBuf.contents;
        // NSLog(@"out[0]=(%f,%f) out[1]=(%f,%f) out[2]=(%f,%f)",
        //     re[0], im[0], re[1], im[1], re[2], im[2]);
        // std::cerr << "GPU kron result: outId=" << *outId
        //       << " outCoef=(" << outCoef[0] << "," << outCoef[1] << ")\n";
    }
}

extern "C" bool copyGpuBufferToHostFloat(void* gpuHandle, float* dst, size_t count) {
    if (!gpuHandle || !dst) return false;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)gpuHandle;
    if (!buf.contents) return false;
    std::memcpy(dst, buf.contents, count * sizeof(float));
    return true;
}


void releaseGpuBuffer(void* p) {
    if (!p) return;
#if __has_feature(objc_arc)
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)p;
    (void)buf;
#else
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)p;
    [buf release];
#endif
}