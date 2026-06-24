#ifndef HEIAN_CUH
#define HEIAN_CUH

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <stdexcept>

#include "Heian.h"
#include "backend.hpp"

namespace heian_cuda {

constexpr uint32_t KIND_TERMINAL = 0; // SonKind::Terminal
constexpr uint32_t KIND_QMDD     = 1; // SonKind::QMDDNode
constexpr uint32_t KIND_SV       = 2; // SonKind::SVLeaf

struct GPUInputHeaderCUDA {
    uint32_t kind;
    double root_re;
    double root_im;
    uint32_t dim;
};

struct Cx {
    double re;
    double im;
};

inline GPUInputHeaderCUDA makeHeader(const GPUInput& in) {
    GPUInputHeaderCUDA h{};
    h.kind = static_cast<uint32_t>(in.kind);
    h.root_re = static_cast<double>(in.root_re);
    h.root_im = static_cast<double>(in.root_im);
    h.dim = static_cast<uint32_t>(in.dim);
    return h;
}

#define CUDA_CHECK(expr) do {                                  \
    cudaError_t _e = (expr);                                   \
    if (_e != cudaSuccess) {                                   \
        throw std::runtime_error(cudaGetErrorString(_e));      \
    }                                                          \
} while (0)

__device__ Cx cmul(Cx a, Cx b);
__device__ Cx cadd(Cx a, Cx b);
__device__ bool isNonZero(double re, double im);
__device__ uint64_t fnv1a64_step(uint64_t h, uint64_t v);

__device__ Cx evalDD(
    const GPUEdge* edges,
    Cx rootW,
    uint32_t row,
    uint32_t col,
    uint32_t dim
);

__device__ Cx evalInput(
    const GPUInputHeaderCUDA& hdr,
    const GPUEdge* edges,
    const double* inRe,
    const double* inIm,
    uint32_t row,
    uint32_t col,
    uint32_t tid,
    bool applyRoot
);

// kernels
__global__ void mul_any2_kernel(
    GPUInputHeaderCUDA hdrA, GPUInputHeaderCUDA hdrB,
    const GPUEdge* edgesA, const GPUEdge* edgesB,
    const double* inReA, const double* inImA,
    const double* inReB, const double* inImB,
    double* outRe, double* outIm
);

__global__ void add_any2_kernel(
    GPUInputHeaderCUDA hdrA, GPUInputHeaderCUDA hdrB,
    const GPUEdge* edgesA, const GPUEdge* edgesB,
    const double* inReA, const double* inImA,
    const double* inReB, const double* inImB,
    double* outRe, double* outIm
);

__global__ void kron_any2_kernel(
    GPUInputHeaderCUDA hdrA, GPUInputHeaderCUDA hdrB,
    const GPUEdge* edgesA, const GPUEdge* edgesB,
    const double* inReA, const double* inImA,
    const double* inReB, const double* inImB,
    double* outRe, double* outIm
);

__global__ void find_first_nonzero_kernel(
    const double* re, const double* im, uint32_t total, unsigned int* firstIdx
);

__global__ void write_coef_kernel(
    const double* re, const double* im,
    const unsigned int* firstIdx,
    double* coefRe, double* coefIm
);

__global__ void norm_apply_kernel(
    double* re, double* im, uint32_t total,
    const unsigned int* firstIdx,
    const double* coefRe, const double* coefIm
);

__global__ void hash_serial_kernel(
    const double* re, const double* im, uint32_t total, uint64_t* outId
);

// host helpers
GPUEdge* toDeviceEdges(const std::vector<GPUEdge>& edges);

void run_postprocess(
    double* dOutRe,
    double* dOutIm,
    uint32_t total,
    int64_t* outId,
    double* outCoef
);

} // namespace heian_cuda

#endif // __CUDACC__
#endif