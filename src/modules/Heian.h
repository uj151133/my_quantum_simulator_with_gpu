#ifndef HEIAN_H
#define HEIAN_H

#include <cstddef>
#include <cstdint>
#include "backend.hpp"

extern "C" void runMulAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    float* outRe, size_t out_bytes,
    float* outIm, size_t out_bytes2,
    int64_t* outId,
    float* outCoef
);

extern "C" void runAddAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    float* outRe, size_t out_bytes,
    float* outIm, size_t out_bytes2,
    int64_t* outId,
    float* outCoef
);

extern "C" void runKronAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    float* outRe, size_t out_bytes,
    float* outIm, size_t out_bytes2,
    int64_t* outId,
    float* outCoef
);

#endif