#ifndef HEIAN_H
#define HEIAN_H

#include <cstddef>
#include <cstdint>
#include "backend.hpp"

extern "C" void runMulAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    void** outRe,
    void** outIm,
    int64_t* outId,
    float* outCoef
);

extern "C" void runAddAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    void** outRe,
    void** outIm,
    int64_t* outId,
    float* outCoef
);

extern "C" void runKronAny2Wrapper(
    const GPUInput& A,
    const GPUInput& B,
    void** outRe,
    void** outIm,
    int64_t* outId,
    float* outCoef
);

#endif