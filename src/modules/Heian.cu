#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <limits>
#include <cstring>
#include <stdexcept>

#include "Heian.h"
#include "backend.hpp"
#include "../models/sv.hpp"

namespace {

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

__device__ inline Cx cmul(Cx a, Cx b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

__device__ inline Cx cadd(Cx a, Cx b) {
    return {a.re + b.re, a.im + b.im};
}

__device__ inline bool isNonZero(double re, double im) {
    return (re != 0.0) || (im != 0.0);
}

__device__ inline uint64_t fnv1a64_step(uint64_t h, uint64_t v) {
    h ^= v;
    h *= 1099511628211ull;
    return h;
}

__device__ Cx evalDD(
    const GPUEdge* edges,
    Cx rootW,
    uint32_t row,
    uint32_t col,
    uint32_t dim
) {
    uint32_t levels = 0;
    uint32_t t = dim;
    while (t > 1) { t >>= 1; ++levels; }

    Cx acc = rootW;
    int node = 0;

    for (uint32_t level = 0; level < levels; ++level) {
        uint32_t shift = (levels - 1 - level);
        uint32_t rb = (row >> shift) & 1u;
        uint32_t cb = (col >> shift) & 1u;
        uint32_t k  = (rb << 1) | cb;

        const GPUEdge e = edges[node * 4 + k];
        acc = cmul(acc, {static_cast<double>(e.re), static_cast<double>(e.im)});

        if (e.childIndex == -1) break;
        node = e.childIndex;
    }

    return acc;
}

__device__ Cx evalInput(
    const GPUInputHeaderCUDA& hdr,
    const GPUEdge* edges,
    const double* inRe,
    const double* inIm,
    uint32_t row,
    uint32_t col,
    uint32_t tid
) {
    if (hdr.kind == KIND_SV) {
        Cx v{inRe[tid], inIm[tid]};
        return cmul({hdr.root_re, hdr.root_im}, v);
    } else if (hdr.kind == KIND_QMDD) {
        return evalDD(edges, {hdr.root_re, hdr.root_im}, row, col, hdr.dim);
    }
    return {0.0, 0.0};
}

__global__ void mul_any2_kernel(
    GPUInputHeaderCUDA hdrA,
    GPUInputHeaderCUDA hdrB,
    const GPUEdge* edgesA,
    const GPUEdge* edgesB,
    const double* inReA,
    const double* inImA,
    const double* inReB,
    const double* inImB,
    double* outRe,
    double* outIm
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dim = hdrB.dim;
    uint32_t total = dim * dim;
    if (tid >= total) return;

    uint32_t row = tid / dim;
    uint32_t col = tid - row * dim;

    Cx acc{0.0, 0.0};
    for (uint32_t k = 0; k < dim; ++k) {
        Cx a = evalInput(hdrA, edgesA, inReA, inImA, row, k, row * dim + k);
        Cx b = evalInput(hdrB, edgesB, inReB, inImB, k, col, k * dim + col);
        acc = cadd(acc, cmul(a, b));
    }

    outRe[tid] = acc.re;
    outIm[tid] = acc.im;
}

__global__ void add_any2_kernel(
    GPUInputHeaderCUDA hdrA,
    GPUInputHeaderCUDA hdrB,
    const GPUEdge* edgesA,
    const GPUEdge* edgesB,
    const double* inReA,
    const double* inImA,
    const double* inReB,
    const double* inImB,
    double* outRe,
    double* outIm
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dim = hdrB.dim;
    uint32_t total = dim * dim;
    if (tid >= total) return;

    uint32_t row = tid / dim;
    uint32_t col = tid - row * dim;

    Cx a = evalInput(hdrA, edgesA, inReA, inImA, row, col, tid);
    Cx b = evalInput(hdrB, edgesB, inReB, inImB, row, col, tid);
    Cx v = cadd(a, b);

    outRe[tid] = v.re;
    outIm[tid] = v.im;
}

__global__ void kron_any2_kernel(
    GPUInputHeaderCUDA hdrA,
    GPUInputHeaderCUDA hdrB,
    const GPUEdge* edgesA,
    const GPUEdge* edgesB,
    const double* inReA,
    const double* inImA,
    const double* inReB,
    const double* inImB,
    double* outRe,
    double* outIm
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t dimA = hdrA.dim;
    uint32_t dimB = hdrB.dim;
    uint32_t dimOut = dimA * dimB;
    uint32_t total = dimOut * dimOut;
    if (tid >= total) return;

    uint32_t row = tid / dimOut;
    uint32_t col = tid - row * dimOut;

    uint32_t rowA = row / dimB;
    uint32_t colA = col / dimB;
    uint32_t rowB = row % dimB;
    uint32_t colB = col % dimB;

    Cx a = evalInput(hdrA, edgesA, inReA, inImA, rowA, colA, rowA * dimA + colA);
    Cx b = evalInput(hdrB, edgesB, inReB, inImB, rowB, colB, rowB * dimB + colB);
    Cx v = cmul(a, b);

    outRe[tid] = v.re;
    outIm[tid] = v.im;
}

__global__ void find_first_nonzero_kernel(
    const double* re,
    const double* im,
    uint32_t total,
    unsigned int* firstIdx
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    if (isNonZero(re[tid], im[tid])) {
        atomicMin(firstIdx, tid);
    }
}

__global__ void write_coef_kernel(
    const double* re,
    const double* im,
    const unsigned int* firstIdx,
    double* coefRe,
    double* coefIm
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int idx = firstIdx[0];
    if (idx == 0xFFFFFFFFu) {
        coefRe[0] = 1.0;
        coefIm[0] = 0.0;
    } else {
        coefRe[0] = re[idx];
        coefIm[0] = im[idx];
    }
}

__global__ void norm_apply_kernel(
    double* re,
    double* im,
    uint32_t total,
    const unsigned int* firstIdx,
    const double* coefRe,
    const double* coefIm
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    if (firstIdx[0] == 0xFFFFFFFFu) return;

    double cr = coefRe[0];
    double ci = coefIm[0];
    double denom = cr * cr + ci * ci;

    double vr = re[tid];
    double vi = im[tid];

    double nr = (vr * cr + vi * ci) / denom;
    double ni = (vi * cr - vr * ci) / denom;

    re[tid] = nr;
    im[tid] = ni;
}

__global__ void hash_serial_kernel(
    const double* re,
    const double* im,
    uint32_t total,
    uint64_t* outId
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < total; ++i) {
        uint64_t rb = static_cast<uint64_t>(__double_as_longlong(re[i]));
        uint64_t ib = static_cast<uint64_t>(__double_as_longlong(im[i]));
        h = fnv1a64_step(h, rb);
        h = fnv1a64_step(h, ib);
    }
    outId[0] = h;
}

inline GPUEdge* toDeviceEdges(const std::vector<GPUEdge>& edges) {
    if (edges.empty()) return nullptr;
    GPUEdge* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, edges.size() * sizeof(GPUEdge)));
    CUDA_CHECK(cudaMemcpy(d, edges.data(), edges.size() * sizeof(GPUEdge), cudaMemcpyHostToDevice));
    return d;
}

inline void run_postprocess(
    double* dOutRe,
    double* dOutIm,
    uint32_t total,
    int64_t* outId,
    double* outCoef
) {
    unsigned int* dFirst = nullptr;
    double* dCoefRe = nullptr;
    double* dCoefIm = nullptr;
    uint64_t* dHash = nullptr;

    CUDA_CHECK(cudaMalloc(&dFirst, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dCoefRe, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCoefIm, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dHash, sizeof(uint64_t)));

    unsigned int init = 0xFFFFFFFFu;
    CUDA_CHECK(cudaMemcpy(dFirst, &init, sizeof(unsigned int), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);

    find_first_nonzero_kernel<<<grid, block>>>(dOutRe, dOutIm, total, dFirst);
    write_coef_kernel<<<1, 1>>>(dOutRe, dOutIm, dFirst, dCoefRe, dCoefIm);
    norm_apply_kernel<<<grid, block>>>(dOutRe, dOutIm, total, dFirst, dCoefRe, dCoefIm);
    hash_serial_kernel<<<1, 1>>>(dOutRe, dOutIm, total, dHash);

    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t h = 0;
    CUDA_CHECK(cudaMemcpy(&outCoef[0], dCoefRe, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&outCoef[1], dCoefIm, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h, dHash, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    *outId = static_cast<int64_t>(h);

    cudaFree(dFirst);
    cudaFree(dCoefRe);
    cudaFree(dCoefIm);
    cudaFree(dHash);
}

} // namespace

extern "C" void runMulAny2Wrapper(
    const GPUInput& A, const GPUInput& B,
    void** outRe, void** outIm, int64_t* outId, gpu_precision* outCoef
) {
    GPUInputHeaderCUDA hdrA = makeHeader(A);
    GPUInputHeaderCUDA hdrB = makeHeader(B);

    const uint32_t dim = hdrA.dim;
    const uint32_t total = dim * dim;
    const size_t bytes = static_cast<size_t>(total) * sizeof(double);

    auto* dInReA = (A.kind == SonKind::SVLeaf) ? static_cast<const double*>(A.sv.reHandle) : nullptr;
    auto* dInImA = (A.kind == SonKind::SVLeaf) ? static_cast<const double*>(A.sv.imHandle) : nullptr;
    auto* dInReB = (B.kind == SonKind::SVLeaf) ? static_cast<const double*>(B.sv.reHandle) : nullptr;
    auto* dInImB = (B.kind == SonKind::SVLeaf) ? static_cast<const double*>(B.sv.imHandle) : nullptr;

    GPUEdge* dEdgesA = (A.kind == SonKind::QMDDNode) ? toDeviceEdges(A.qmdd.edges) : nullptr;
    GPUEdge* dEdgesB = (B.kind == SonKind::QMDDNode) ? toDeviceEdges(B.qmdd.edges) : nullptr;

    double* dOutRe = nullptr;
    double* dOutIm = nullptr;
    CUDA_CHECK(cudaMalloc(&dOutRe, bytes));
    CUDA_CHECK(cudaMalloc(&dOutIm, bytes));

    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);

    mul_any2_kernel<<<grid, block>>>(
        hdrA, hdrB, dEdgesA, dEdgesB,
        dInReA, dInImA, dInReB, dInImB,
        dOutRe, dOutIm
    );

    double coef[2] = {0.0, 0.0};
    run_postprocess(dOutRe, dOutIm, total, outId, coef);

    // Metal実装に合わせて rootA*rootB を掛ける
    double rA = static_cast<double>(A.root_re), iA = static_cast<double>(A.root_im);
    double rB = static_cast<double>(B.root_re), iB = static_cast<double>(B.root_im);
    double r = rA * rB - iA * iB;
    double i = rA * iB + iA * rB;
    outCoef[0] = coef[0] * r - coef[1] * i;
    outCoef[1] = coef[0] * i + coef[1] * r;

    if (outRe) *outRe = dOutRe;
    if (outIm) *outIm = dOutIm;

    if (dEdgesA) cudaFree(dEdgesA);
    if (dEdgesB) cudaFree(dEdgesB);
}

extern "C" void runAddAny2Wrapper(
    const GPUInput& A, const GPUInput& B,
    void** outRe, void** outIm, int64_t* outId, gpu_precision* outCoef
) {
    GPUInputHeaderCUDA hdrA = makeHeader(A);
    GPUInputHeaderCUDA hdrB = makeHeader(B);

    const uint32_t dim = hdrA.dim;
    const uint32_t total = dim * dim;
    const size_t bytes = static_cast<size_t>(total) * sizeof(double);

    auto* dInReA = (A.kind == SonKind::SVLeaf) ? static_cast<const double*>(A.sv.reHandle) : nullptr;
    auto* dInImA = (A.kind == SonKind::SVLeaf) ? static_cast<const double*>(A.sv.imHandle) : nullptr;
    auto* dInReB = (B.kind == SonKind::SVLeaf) ? static_cast<const double*>(B.sv.reHandle) : nullptr;
    auto* dInImB = (B.kind == SonKind::SVLeaf) ? static_cast<const double*>(B.sv.imHandle) : nullptr;

    GPUEdge* dEdgesA = (A.kind == SonKind::QMDDNode) ? toDeviceEdges(A.qmdd.edges) : nullptr;
    GPUEdge* dEdgesB = (B.kind == SonKind::QMDDNode) ? toDeviceEdges(B.qmdd.edges) : nullptr;

    double* dOutRe = nullptr;
    double* dOutIm = nullptr;
    CUDA_CHECK(cudaMalloc(&dOutRe, bytes));
    CUDA_CHECK(cudaMalloc(&dOutIm, bytes));

    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);

    add_any2_kernel<<<grid, block>>>(
        hdrA, hdrB, dEdgesA, dEdgesB,
        dInReA, dInImA, dInReB, dInImB,
        dOutRe, dOutIm
    );

    double coef[2] = {0.0, 0.0};
    run_postprocess(dOutRe, dOutIm, total, outId, coef);
    outCoef[0] = coef[0];
    outCoef[1] = coef[1];

    if (outRe) *outRe = dOutRe;
    if (outIm) *outIm = dOutIm;

    if (dEdgesA) cudaFree(dEdgesA);
    if (dEdgesB) cudaFree(dEdgesB);
}

extern "C" void runKronAny2Wrapper(
    const GPUInput& A, const GPUInput& B,
    void** outRe, void** outIm, int64_t* outId, gpu_precision* outCoef
) {
    GPUInputHeaderCUDA hdrA = makeHeader(A);
    GPUInputHeaderCUDA hdrB = makeHeader(B);

    const uint32_t dimOut = hdrA.dim * hdrB.dim;
    const uint32_t total = dimOut * dimOut;
    const size_t bytes = static_cast<size_t>(total) * sizeof(double);

    auto* dInReA = (A.kind == SonKind::SVLeaf) ? static_cast<const double*>(A.sv.reHandle) : nullptr;
    auto* dInImA = (A.kind == SonKind::SVLeaf) ? static_cast<const double*>(A.sv.imHandle) : nullptr;
    auto* dInReB = (B.kind == SonKind::SVLeaf) ? static_cast<const double*>(B.sv.reHandle) : nullptr;
    auto* dInImB = (B.kind == SonKind::SVLeaf) ? static_cast<const double*>(B.sv.imHandle) : nullptr;

    GPUEdge* dEdgesA = (A.kind == SonKind::QMDDNode) ? toDeviceEdges(A.qmdd.edges) : nullptr;
    GPUEdge* dEdgesB = (B.kind == SonKind::QMDDNode) ? toDeviceEdges(B.qmdd.edges) : nullptr;

    double* dOutRe = nullptr;
    double* dOutIm = nullptr;
    CUDA_CHECK(cudaMalloc(&dOutRe, bytes));
    CUDA_CHECK(cudaMalloc(&dOutIm, bytes));

    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);

    kron_any2_kernel<<<grid, block>>>(
        hdrA, hdrB, dEdgesA, dEdgesB,
        dInReA, dInImA, dInReB, dInImB,
        dOutRe, dOutIm
    );

    double coef[2] = {0.0, 0.0};
    run_postprocess(dOutRe, dOutIm, total, outId, coef);

    // Metal実装に合わせて rootA*rootB を掛ける
    double rA = static_cast<double>(A.root_re), iA = static_cast<double>(A.root_im);
    double rB = static_cast<double>(B.root_re), iB = static_cast<double>(B.root_im);
    double r = rA * rB - iA * iB;
    double i = rA * iB + iA * rB;
    outCoef[0] = coef[0] * r - coef[1] * i;
    outCoef[1] = coef[0] * i + coef[1] * r;

    if (outRe) *outRe = dOutRe;
    if (outIm) *outIm = dOutIm;

    if (dEdgesA) cudaFree(dEdgesA);
    if (dEdgesB) cudaFree(dEdgesB);
}

extern "C" void releaseGpuBuffer(void* p) {
    if (!p) return;
    cudaFree(p);
}

// 既存コード互換（qmdd.cpp が float で読んでいるため）
extern "C" bool copyGpuBufferToHostFloat(void* gpuHandle, float* dst, size_t count) {
    if (!gpuHandle || !dst) return false;
    std::vector<double> tmp(count, 0.0);
    if (cudaMemcpy(tmp.data(), gpuHandle, count * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (size_t i = 0; i < count; ++i) dst[i] = static_cast<float>(tmp[i]);
    return true;
}

// double版（将来的にこちらへ寄せる）
extern "C" bool copyGpuBufferToHostDouble(void* gpuHandle, double* dst, size_t count) {
    if (!gpuHandle || !dst) return false;
    return cudaMemcpy(dst, gpuHandle, count * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
}