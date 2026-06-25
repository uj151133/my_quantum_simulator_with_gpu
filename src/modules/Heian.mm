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

// === 使い回しスクラッチ（OSスレッドごと）=========================================
// Metal も per-op の newBuffer をやめ、固定サイズの一時バッファは使い回す。
// fiber が複数スレッドで走るので thread_local（共有するとデータ競合）。各 op は
// waitUntilCompleted まで同期するので、同一スレッド内での使い回しは安全。
// ARC で消えないよう strong 参照（id）で保持する。
struct MtlScratch {
    id<MTLBuffer> outId   = nil;   // sizeof(uint64_t)
    id<MTLBuffer> outCoef = nil;   // sizeof(float)*2
};
static thread_local MtlScratch g_mtl;

// 固定サイズの共有バッファを「無ければ1回だけ作って」使い回す
static id<MTLBuffer> ensureSharedBuf(id<MTLDevice> dev, id<MTLBuffer>& slot, NSUInteger len) {
    if (slot == nil) {
        slot = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
    }
    return slot;
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

static void runNormalizePass(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> outReBuf,
    id<MTLBuffer> outImBuf,
    id<MTLBuffer> outCoefBuf,
    uint32_t total
) {
    id<MTLFunction> p1Fn = [library newFunctionWithName:@"norm_pass1"];
    id<MTLFunction> p2Fn = [library newFunctionWithName:@"norm_pass2"];
    id<MTLFunction> p3Fn = [library newFunctionWithName:@"norm_apply"];

    id<MTLComputePipelineState> p1PSO = getPSO(device, library, @"norm_pass1");
    id<MTLComputePipelineState> p2PSO = getPSO(device, library, @"norm_pass2");
    id<MTLComputePipelineState> p3PSO = getPSO(device, library, @"norm_apply");

    NSUInteger tg = p1PSO.maxTotalThreadsPerThreadgroup;
    NSUInteger groups = (total + tg - 1) / tg;

    id<MTLBuffer> groupIdxBuf =
        [device newBufferWithLength:sizeof(uint32_t)*groups options:MTLResourceStorageModeShared];
    id<MTLBuffer> groupValBuf =
        [device newBufferWithLength:sizeof(float)*2*groups options:MTLResourceStorageModeShared];

    id<MTLBuffer> totalBuf =
        [device newBufferWithBytes:&total length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> groupBuf =
        [device newBufferWithBytes:&groups length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    id<MTLBuffer> idxBuf =
        [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p1PSO];
        [enc setBuffer:outReBuf offset:0 atIndex:0];
        [enc setBuffer:outImBuf offset:0 atIndex:1];
        [enc setBuffer:groupIdxBuf offset:0 atIndex:2];
        [enc setBuffer:groupValBuf offset:0 atIndex:3];
        [enc setBuffer:totalBuf offset:0 atIndex:4];

        MTLSize grid = MTLSizeMake(groups * tg, 1, 1);
        MTLSize tgs = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
    }

    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p2PSO];
        [enc setBuffer:groupIdxBuf offset:0 atIndex:0];
        [enc setBuffer:groupValBuf offset:0 atIndex:1];
        [enc setBuffer:idxBuf offset:0 atIndex:2];
        [enc setBuffer:outCoefBuf offset:0 atIndex:3];
        [enc setBuffer:groupBuf offset:0 atIndex:4];

        MTLSize grid = MTLSizeMake(1, 1, 1);
        MTLSize tgs = MTLSizeMake(1, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
    }

    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p3PSO];
        [enc setBuffer:outReBuf offset:0 atIndex:0];
        [enc setBuffer:outImBuf offset:0 atIndex:1];
        [enc setBuffer:idxBuf offset:0 atIndex:2];
        [enc setBuffer:outCoefBuf offset:0 atIndex:3];
        [enc setBuffer:totalBuf offset:0 atIndex:4];

        MTLSize grid = MTLSizeMake(total, 1, 1);
        MTLSize tgs = MTLSizeMake(p3PSO.maxTotalThreadsPerThreadgroup, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
    }
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
    id<MTLBuffer> outIdBuf,
    id<MTLBuffer> outCoefBuf
) {
    uint32_t dim = hdrA.dim;          // A,B 同 dim 前提
    uint32_t total = dim * dim;
    size_t denseBytes = (size_t)total * sizeof(float);

    id<MTLBuffer> dimBuf = [device newBufferWithBytes:&dim length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdrBufA = [device newBufferWithBytes:&hdrA length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdrBufB = [device newBufferWithBytes:&hdrB length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];
    id<MTLBuffer> edgesBufA = edgesA_bytes ? [device newBufferWithBytes:edgesA length:edgesA_bytes options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> edgesBufB = edgesB_bytes ? [device newBufferWithBytes:edgesB length:edgesB_bytes options:MTLResourceStorageModeShared] : nil;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];

    // --- Stage 1: QMDD オペランドを dense 化（SV は既に dense なのでそのまま使う）---
    id<MTLComputePipelineState> matPSO = getPSO(device, library, @"materialize_any");
    NSUInteger matTG = matPSO.maxTotalThreadsPerThreadgroup;
    MTLSize matGrid = MTLSizeMake(total, 1, 1);
    MTLSize matTgs  = MTLSizeMake(matTG, 1, 1);

    id<MTLBuffer> aRe = inReBufA, aIm = inImBufA;   // SV ならハンドルをそのまま
    id<MTLBuffer> bRe = inReBufB, bIm = inImBufB;

    if (hdrA.kind == 1u) {  // QMDDNode → dense 化
        aRe = [device newBufferWithLength:denseBytes options:MTLResourceStorageModeShared];
        aIm = [device newBufferWithLength:denseBytes options:MTLResourceStorageModeShared];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:matPSO];
        [enc setBuffer:hdrBufA   offset:0 atIndex:0];
        [enc setBuffer:edgesBufA offset:0 atIndex:1];
        [enc setBuffer:inReBufA  offset:0 atIndex:2];
        [enc setBuffer:inImBufA  offset:0 atIndex:3];
        [enc setBuffer:aRe       offset:0 atIndex:4];
        [enc setBuffer:aIm       offset:0 atIndex:5];
        [enc setBuffer:dimBuf    offset:0 atIndex:6];
        [enc dispatchThreads:matGrid threadsPerThreadgroup:matTgs];
        [enc endEncoding];
    }
    if (hdrB.kind == 1u) {  // QMDDNode → dense 化
        bRe = [device newBufferWithLength:denseBytes options:MTLResourceStorageModeShared];
        bIm = [device newBufferWithLength:denseBytes options:MTLResourceStorageModeShared];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:matPSO];
        [enc setBuffer:hdrBufB   offset:0 atIndex:0];
        [enc setBuffer:edgesBufB offset:0 atIndex:1];
        [enc setBuffer:inReBufB  offset:0 atIndex:2];
        [enc setBuffer:inImBufB  offset:0 atIndex:3];
        [enc setBuffer:bRe       offset:0 atIndex:4];
        [enc setBuffer:bIm       offset:0 atIndex:5];
        [enc setBuffer:dimBuf    offset:0 atIndex:6];
        [enc dispatchThreads:matGrid threadsPerThreadgroup:matTgs];
        [enc endEncoding];
    }

    // --- mul_any2 を「両方 SV 扱い（dense）」で実行：evalDD 無しの素直な dense matmul ---
    GPUInputHeader svHdr; svHdr.kind = 2u; svHdr.root_re = 1.0f; svHdr.root_im = 0.0f; svHdr.dim = dim;
    id<MTLBuffer> svHdrBuf = [device newBufferWithBytes:&svHdr length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];

    id<MTLComputePipelineState> pso = getPSO(device, library, @"mul_any2");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:svHdrBuf  offset:0 atIndex:0];
    [enc setBuffer:svHdrBuf  offset:0 atIndex:1];
    [enc setBuffer:edgesBufA offset:0 atIndex:2];   // kind=SV なので未使用
    [enc setBuffer:edgesBufB offset:0 atIndex:3];   // 同上
    [enc setBuffer:aRe       offset:0 atIndex:4];
    [enc setBuffer:aIm       offset:0 atIndex:5];
    [enc setBuffer:bRe       offset:0 atIndex:6];
    [enc setBuffer:bIm       offset:0 atIndex:7];
    [enc setBuffer:outReBuf  offset:0 atIndex:8];
    [enc setBuffer:outImBuf  offset:0 atIndex:9];

    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    MTLSize tgs = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    [enc endEncoding];

    runNormalizePass(device, library, cmd, outReBuf, outImBuf, outCoefBuf, total);
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
    id<MTLBuffer> outIdBuf,
    id<MTLBuffer> outCoefBuf
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

    runNormalizePass(device, library, cmd, outReBuf, outImBuf, outCoefBuf, total);
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
    id<MTLBuffer> outIdBuf,
    id<MTLBuffer> outCoefBuf
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

    runNormalizePass(device, library, cmd, outReBuf, outImBuf, outCoefBuf, total);
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

        id<MTLBuffer> outIdBuf = ensureSharedBuf(ctx.device, g_mtl.outId, sizeof(uint64_t));

        id<MTLBuffer> outCoefBuf = ensureSharedBuf(ctx.device, g_mtl.outCoef, sizeof(float)*2);


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
            outIdBuf,
            outCoefBuf
        );

        if (outRe) { [outReBuf retain]; *outRe = (__bridge void*)outReBuf; }
        if (outIm) { [outImBuf retain]; *outIm = (__bridge void*)outImBuf; }

        if (outId) {
            *outId = *(int64_t*)outIdBuf.contents;
        }

        if (outCoef) {
            float* c = (float*)outCoefBuf.contents;
            float coefRe = c[0];
            float coefIm = c[1];

            float rA = (float)A.root_re, iA = (float)A.root_im;
            float rB = (float)B.root_re, iB = (float)B.root_im;
            float r = rA*rB - iA*iB;
            float i = rA*iB + iA*rB;

            // coef * (rootA*rootB)
            outCoef[0] = coefRe * r - coefIm * i;
            outCoef[1] = coefRe * i + coefIm * r;
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

        id<MTLBuffer> outIdBuf = ensureSharedBuf(ctx.device, g_mtl.outId, sizeof(uint64_t));

        id<MTLBuffer> outCoefBuf = ensureSharedBuf(ctx.device, g_mtl.outCoef, sizeof(float)*2);

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
            outIdBuf,
            outCoefBuf
        );

        if (outRe) { [outReBuf retain]; *outRe = (__bridge void*)outReBuf; }
        if (outIm) { [outImBuf retain]; *outIm = (__bridge void*)outImBuf; }

        if (outId) {
            *outId = *(int64_t*)outIdBuf.contents;
        }

        if (outCoef) {
            float* c = (float*)outCoefBuf.contents;
            outCoef[0] = c[0];
            outCoef[1] = c[1];
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

        id<MTLBuffer> outIdBuf = ensureSharedBuf(ctx.device, g_mtl.outId, sizeof(uint64_t));

        id<MTLBuffer> outCoefBuf = ensureSharedBuf(ctx.device, g_mtl.outCoef, sizeof(float)*2);

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
            outIdBuf,
            outCoefBuf
        );

        if (outRe) { [outReBuf retain]; *outRe = (__bridge void*)outReBuf; }
        if (outIm) { [outImBuf retain]; *outIm = (__bridge void*)outImBuf; }

        if (outId) {
            *outId = *(int64_t*)outIdBuf.contents;
        }

        if (outCoef) {
            float* c = (float*)outCoefBuf.contents;
            float coefRe = c[0];
            float coefIm = c[1];

            float rA = (float)A.root_re, iA = (float)A.root_im;
            float rB = (float)B.root_re, iB = (float)B.root_im;
            float r = rA*rB - iA*iB;
            float i = rA*iB + iA*rB;

            // coef * (rootA*rootB)
            outCoef[0] = coefRe * r - coefIm * i;
            outCoef[1] = coefRe * i + coefIm * r;
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