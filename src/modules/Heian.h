#ifndef HEIAN_H
#define HEIAN_H

#include <cstddef>
#include <cstdint>
#include "backend.hpp"

#ifdef __cplusplus
extern "C" {
#endif
void runMulAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    void** outRe,
    void** outIm,
    int64_t* outId,
    gpu_precision* outCoef
);
void runAddAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    void** outRe,
    void** outIm,
    int64_t* outId,
    gpu_precision* outCoef
);
void runKronAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    void** outRe,
    void** outIm,
    int64_t* outId,
    gpu_precision* outCoef
);

void releaseGpuBuffer(void* p);

bool copyGpuBufferToHostFloat(void* gpuHandle, float* dst, size_t count);

bool copyGpuBufferToHostDouble(void* gpuHandle, double* dst, size_t count);
#ifdef __cplusplus
}
#endif

#endif