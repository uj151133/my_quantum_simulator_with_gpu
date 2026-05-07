#include <cuda_runtime.h>
#include "../models/sv.hpp"

void releaseGpuBuffer(void* p) {
    if (!p) return;
    cudaFree(p);
}