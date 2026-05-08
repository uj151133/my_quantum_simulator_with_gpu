#import <Metal/Metal.h>
#include "../models/sv.hpp"
#include "../models/qmdd.hpp"
#include "backend.hpp"

struct GPUInputHeader {
    uint32_t kind;    // 0=QMDD, 1=SV, 2=Terminal
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

    id<MTLComputePipelineState> p1PSO = [device newComputePipelineStateWithFunction:p1Fn error:nil];
    id<MTLComputePipelineState> p2PSO = [device newComputePipelineStateWithFunction:p2Fn error:nil];
    id<MTLComputePipelineState> p3PSO = [device newComputePipelineStateWithFunction:p3Fn error:nil];

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

    // pass1
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

    // pass2
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

    // pass3
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
    id<MTLComputePipelineState> p1PSO = [device newComputePipelineStateWithFunction:p1Fn error:nil];
    id<MTLComputePipelineState> p2PSO = [device newComputePipelineStateWithFunction:p2Fn error:nil];

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
    const GPUInputHeader& hdrA,
    const GPUInputHeader& hdrB,
    const void* edgesA, size_t edgesA_bytes,
    const void* edgesB, size_t edgesB_bytes,
    const float* inReA, size_t inA_bytes,
    const float* inImA, size_t inA_bytes2,
    const float* inReB, size_t inB_bytes,
    const float* inImB, size_t inB_bytes2,
    float* outRe, size_t out_bytes,
    float* outIm, size_t out_bytes2,
    id<MTLBuffer> outIdBuf,
    id<MTLBuffer> outCoefBuf
) {
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:@"mul_any2"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];

    id<MTLBuffer> hdrBufA = [device newBufferWithBytes:&hdrA length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdrBufB = [device newBufferWithBytes:&hdrB length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];

    id<MTLBuffer> edgesBufA = edgesA_bytes ? [device newBufferWithBytes:edgesA length:edgesA_bytes options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> edgesBufB = edgesB_bytes ? [device newBufferWithBytes:edgesB length:edgesB_bytes options:MTLResourceStorageModeShared] : nil;

    id<MTLBuffer> inReBufA = [device newBufferWithBytes:inReA length:inA_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> inImBufA = [device newBufferWithBytes:inImA length:inA_bytes2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> inReBufB = [device newBufferWithBytes:inReB length:inB_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> inImBufB = [device newBufferWithBytes:inImB length:inB_bytes2 options:MTLResourceStorageModeShared];

    id<MTLBuffer> outReBuf = [device newBufferWithBytes:outRe length:out_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> outImBuf = [device newBufferWithBytes:outIm length:out_bytes2 options:MTLResourceStorageModeShared];

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

static void runAddAny2(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    const GPUInputHeader& hdrA,
    const GPUInputHeader& hdrB,
    const void* edgesA, size_t edgesA_bytes,
    const void* edgesB, size_t edgesB_bytes,
    const float* inReA, size_t inA_bytes,
    const float* inImA, size_t inA_bytes2,
    const float* inReB, size_t inB_bytes,
    const float* inImB, size_t inB_bytes2,
    float* outRe, size_t out_bytes,
    float* outIm, size_t out_bytes2,
    id<MTLBuffer> outIdBuf,
    id<MTLBuffer> outCoefBuf
) {
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:@"add_any2"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];

    id<MTLBuffer> hdrBufA = [device newBufferWithBytes:&hdrA length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdrBufB = [device newBufferWithBytes:&hdrB length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];

    id<MTLBuffer> edgesBufA = edgesA_bytes ? [device newBufferWithBytes:edgesA length:edgesA_bytes options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> edgesBufB = edgesB_bytes ? [device newBufferWithBytes:edgesB length:edgesB_bytes options:MTLResourceStorageModeShared] : nil;

    id<MTLBuffer> inReBufA = [device newBufferWithBytes:inReA length:inA_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> inImBufA = [device newBufferWithBytes:inImA length:inA_bytes2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> inReBufB = [device newBufferWithBytes:inReB length:inB_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> inImBufB = [device newBufferWithBytes:inImB length:inB_bytes2 options:MTLResourceStorageModeShared];

    id<MTLBuffer> outReBuf = [device newBufferWithBytes:outRe length:out_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> outImBuf = [device newBufferWithBytes:outIm length:out_bytes2 options:MTLResourceStorageModeShared];

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
    const GPUInputHeader& hdrA,
    const GPUInputHeader& hdrB,
    const void* edgesA, size_t edgesA_bytes,
    const void* edgesB, size_t edgesB_bytes,
    const float* inReA, size_t inA_bytes,
    const float* inImA, size_t inA_bytes2,
    const float* inReB, size_t inB_bytes,
    const float* inImB, size_t inB_bytes2,
    float* outRe, size_t out_bytes,
    float* outIm, size_t out_bytes2,
    id<MTLBuffer> outIdBuf,
    id<MTLBuffer> outCoefBuf
) {
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError* error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:@"kron_any2"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];

    id<MTLBuffer> hdrBufA = [device newBufferWithBytes:&hdrA length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hdrBufB = [device newBufferWithBytes:&hdrB length:sizeof(GPUInputHeader) options:MTLResourceStorageModeShared];

    id<MTLBuffer> edgesBufA = edgesA_bytes ? [device newBufferWithBytes:edgesA length:edgesA_bytes options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> edgesBufB = edgesB_bytes ? [device newBufferWithBytes:edgesB length:edgesB_bytes options:MTLResourceStorageModeShared] : nil;

    id<MTLBuffer> inReBufA = inA_bytes ? [device newBufferWithBytes:inReA length:inA_bytes options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> inImBufA = inA_bytes2 ? [device newBufferWithBytes:inImA length:inA_bytes2 options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> inReBufB = inB_bytes ? [device newBufferWithBytes:inReB length:inB_bytes options:MTLResourceStorageModeShared] : nil;
    id<MTLBuffer> inImBufB = inB_bytes2 ? [device newBufferWithBytes:inImB length:inB_bytes2 options:MTLResourceStorageModeShared] : nil;

    id<MTLBuffer> outReBuf = [device newBufferWithBytes:outRe length:out_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> outImBuf = [device newBufferWithBytes:outIm length:out_bytes2 options:MTLResourceStorageModeShared];

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
    float* outRe, size_t out_bytes,
    float* outIm, size_t out_bytes2,
    int64_t* outId,
    float* outCoef
) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library = [device newDefaultLibrary];

    GPUInputHeader hdrA = makeHeader(A);
    GPUInputHeader hdrB = makeHeader(B);

    const void* edgesA = A.qmdd.edges.empty() ? nullptr : A.qmdd.edges.data();
    size_t edgesA_bytes = A.qmdd.edges.size() * sizeof(GPUEdge);

    const void* edgesB = B.qmdd.edges.empty() ? nullptr : B.qmdd.edges.data();
    size_t edgesB_bytes = B.qmdd.edges.size() * sizeof(GPUEdge);

    const float* inReA = (A.kind == SonKind::SVLeaf) ? (const float*)A.sv.reHandle : nullptr;
    const float* inImA = (A.kind == SonKind::SVLeaf) ? (const float*)A.sv.imHandle : nullptr;
    const float* inReB = (B.kind == SonKind::SVLeaf) ? (const float*)B.sv.reHandle : nullptr;
    const float* inImB = (B.kind == SonKind::SVLeaf) ? (const float*)B.sv.imHandle : nullptr;

    size_t inA_bytes = (A.kind == SonKind::SVLeaf) ? A.dim * A.dim * sizeof(float) : 0;
    size_t inB_bytes = (B.kind == SonKind::SVLeaf) ? B.dim * B.dim * sizeof(float) : 0;

    id<MTLBuffer> outIdBuf =
        [device newBufferWithLength:sizeof(uint64_t)
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> outCoefBuf =
        [device newBufferWithLength:sizeof(float)*2
                             options:MTLResourceStorageModeShared];

    runMulAny2(
        device, library,
        hdrA, hdrB,
        edgesA, edgesA_bytes,
        edgesB, edgesB_bytes,
        inReA, inA_bytes,
        inImA, inA_bytes,
        inReB, inB_bytes,
        inImB, inB_bytes,
        outRe, out_bytes,
        outIm, out_bytes2,
        outIdBuf,
        outCoefBuf
    );

    if (outId) {
        *outId = *(int64_t*)outIdBuf.contents;
    }

    if (outCoef) {
        float* c = (float*)outCoefBuf.contents;
        outCoef[0] = c[0];
        outCoef[1] = c[1];
    }
}

extern "C" void runAddAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    float* outRe, size_t out_bytes,
    float* outIm, size_t out_bytes2,
    int64_t* outId,
    float* outCoef
) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library = [device newDefaultLibrary];

    GPUInputHeader hdrA = makeHeader(A);
    GPUInputHeader hdrB = makeHeader(B);

    const void* edgesA = A.qmdd.edges.empty() ? nullptr : A.qmdd.edges.data();
    size_t edgesA_bytes = A.qmdd.edges.size() * sizeof(GPUEdge);

    const void* edgesB = B.qmdd.edges.empty() ? nullptr : B.qmdd.edges.data();
    size_t edgesB_bytes = B.qmdd.edges.size() * sizeof(GPUEdge);

    const float* inReA = (A.kind == SonKind::SVLeaf) ? (const float*)A.sv.reHandle : nullptr;
    const float* inImA = (A.kind == SonKind::SVLeaf) ? (const float*)A.sv.imHandle : nullptr;
    const float* inReB = (B.kind == SonKind::SVLeaf) ? (const float*)B.sv.reHandle : nullptr;
    const float* inImB = (B.kind == SonKind::SVLeaf) ? (const float*)B.sv.imHandle : nullptr;

    size_t inA_bytes = (A.kind == SonKind::SVLeaf) ? A.dim * A.dim * sizeof(float) : 0;
    size_t inB_bytes = (B.kind == SonKind::SVLeaf) ? B.dim * B.dim * sizeof(float) : 0;

    id<MTLBuffer> outIdBuf =
        [device newBufferWithLength:sizeof(uint64_t)
                             options:MTLResourceStorageModeShared];

    id<MTLBuffer> outCoefBuf =
        [device newBufferWithLength:sizeof(float)*2
                             options:MTLResourceStorageModeShared];

    runAddAny2(
        device, library,
        hdrA, hdrB,
        edgesA, edgesA_bytes,
        edgesB, edgesB_bytes,
        inReA, inA_bytes,
        inImA, inA_bytes,
        inReB, inB_bytes,
        inImB, inB_bytes,
        outRe, out_bytes,
        outIm, out_bytes2,
        outIdBuf,
        outCoefBuf
    );

    if (outId) {
        *outId = *(int64_t*)outIdBuf.contents;
    }
    if (outCoef) {
        float* c = (float*)outCoefBuf.contents;
        outCoef[0] = c[0];
        outCoef[1] = c[1];
    }
}

extern "C" void runKronAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    float* outRe, size_t out_bytes,
    float* outIm, size_t out_bytes2,
    int64_t* outId,
    float* outCoef
) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library = [device newDefaultLibrary];

    GPUInputHeader hdrA = makeHeader(A);
    GPUInputHeader hdrB = makeHeader(B);

    const void* edgesA = A.qmdd.edges.empty() ? nullptr : A.qmdd.edges.data();
    size_t edgesA_bytes = A.qmdd.edges.size() * sizeof(GPUEdge);

    const void* edgesB = B.qmdd.edges.empty() ? nullptr : B.qmdd.edges.data();
    size_t edgesB_bytes = B.qmdd.edges.size() * sizeof(GPUEdge);

    const float* inReA = (A.kind == SonKind::SVLeaf) ? (const float*)A.sv.reHandle : nullptr;
    const float* inImA = (A.kind == SonKind::SVLeaf) ? (const float*)A.sv.imHandle : nullptr;
    const float* inReB = (B.kind == SonKind::SVLeaf) ? (const float*)B.sv.reHandle : nullptr;
    const float* inImB = (B.kind == SonKind::SVLeaf) ? (const float*)B.sv.imHandle : nullptr;

    size_t inA_bytes = (A.kind == SonKind::SVLeaf) ? A.dim * A.dim * sizeof(float) : 0;
    size_t inB_bytes = (B.kind == SonKind::SVLeaf) ? B.dim * B.dim * sizeof(float) : 0;

    id<MTLBuffer> outIdBuf =
        [device newBufferWithLength:sizeof(uint64_t)
                             options:MTLResourceStorageModeShared];

    id<MTLBuffer> outCoefBuf =
        [device newBufferWithLength:sizeof(float)*2
                             options:MTLResourceStorageModeShared];

    runKronAny2(
        device, library,
        hdrA, hdrB,
        edgesA, edgesA_bytes,
        edgesB, edgesB_bytes,
        inReA, inA_bytes,
        inImA, inA_bytes,
        inReB, inB_bytes,
        inImB, inB_bytes,
        outRe, out_bytes,
        outIm, out_bytes2,
        outIdBuf,
        outCoefBuf
    );

    if (outId) {
        *outId = *(int64_t*)outIdBuf.contents;
    }

    if (outCoef) {
        float* c = (float*)outCoefBuf.contents;
        outCoef[0] = c[0];
        outCoef[1] = c[1];
    }
}


void releaseGpuBuffer(void* p) {
    if (!p) return;
#if __has_feature(objc_arc)
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)p;
    (void)buf; // ARC解放
#else
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)p;
    [buf release];
#endif
}