#ifndef HEIAN_CUH
#define HEIAN_CUH

// CUDA バックエンドのヘッダ。Heian.cu だけが実装し、CPU 側は USE_CUDA のときだけ見る。
// Metal 版は Heian.h。公開 API（BatchJob と各 Wrapper）は両者で一致させること。
//   公開 API を変えたら Heian.h も直す
//   CUDA 固有部分（GPUInputHeaderCUDA 以下）は Metal 側に対応物がある: Heian.h の GPUInputHeader
#include <cstddef>
#include <cstdint>
#include <vector>
#include "backend.hpp"

// === 公開 API（Heian.h と一致させること）====================================
// 互いに独立な演算をまとめて 1 往復で処理する。
// オペランドのサイズは演算ごとに違ってよい。
//   CUDA  … threadgroup(block)→演算番号の対応表を持つカーネルを 6 本、CUDA Graph で 1 回起動
//   Metal … 同じ対応表で 6 本を 1 エンコーダに積み 1 回 commit（Metal 3 / macOS 13 以降。
//           それ未満か threadgroup 256 を起動できない GPU なら、ジョブごとに積む従来版）
struct BatchJob {
    const GPUInput* A;
    const GPUInput* B;
    // --- 以下は実行後に埋まる ---
    void*    outRe   = nullptr;
    void*    outIm   = nullptr;   // 成分が全ゼロなら nullptr
    int64_t  outId   = 0;
    gpu_precision outCoef[2] = {0, 0};
    uint32_t dim     = 0;         // 結果の一辺（mul/add は A.dim、kron は A.dim*B.dim）
};

#ifdef __cplusplus
extern "C" {
#endif
void runMulAny2Wrapper(
    const GPUInput& A, const GPUInput& B,
    void** outRe, void** outIm, int64_t* outId, gpu_precision* outCoef
);
void runAddAny2Wrapper(
    const GPUInput& A, const GPUInput& B,
    void** outRe, void** outIm, int64_t* outId, gpu_precision* outCoef
);
void runKronAny2Wrapper(
    const GPUInput& A, const GPUInput& B,
    void** outRe, void** outIm, int64_t* outId, gpu_precision* outCoef
);

void runMulBatchWrapper(BatchJob* jobs, int nJobs);
void runAddBatchWrapper(BatchJob* jobs, int nJobs);
void runKronBatchWrapper(BatchJob* jobs, int nJobs);

void releaseGpuBuffer(void* p);

bool copyGpuBufferToHostFloat(void* gpuHandle, float* dst, size_t count);

bool copyGpuBufferToHostDouble(void* gpuHandle, double* dst, size_t count);
#ifdef __cplusplus
}
#endif

// === 以下は CUDA 固有。nvcc でコンパイルするときだけ有効 ======================
// CPU 側（g++）もこのヘッダを include するので、CUDA 構文はここから下に閉じ込める。
#ifdef __CUDACC__

constexpr uint32_t KIND_QMDD     = 1; // SonKind::QMDDNode
constexpr uint32_t KIND_SV       = 2; // SonKind::ArrayLeaf

// Heian.h の GPUInputHeader と役割が同じ（精度と全ゼロの伝え方だけ違う）:
//   CUDA  … ハンドルをそのまま nullptr で渡し、カーネル内で null 判定
//   Metal … hasRe / hasIm フラグで伝える（nil バッファの参照を避けるため）
struct GPUInputHeaderCUDA {
    uint32_t kind;
    double root_re;
    double root_im;
    uint32_t dim;      // 元の（小さい）一辺
    uint32_t pad;      // 単位行列パディング倍率。実効の一辺は dim*pad
};
__device__ __host__ inline uint32_t effDimH(const GPUInputHeaderCUDA& h) { return h.dim * h.pad; }

struct Cx {
    double re;
    double im;
};

// 1 op の結果（係数とハッシュ）をまとめて 1 回で転送するための箱
struct PostOut {
    double   coefRe;
    double   coefIm;
    uint64_t hash;
    uint32_t zeroRe;   // 実部が全要素ゼロなら 1
    uint32_t zeroIm;   // 虚部が全要素ゼロなら 1
};

// L2 正規化後は全要素 <= 1 なので、この絶対値閾値がそのまま相対誤差になる
#ifndef QMDD_GPU_ZERO_EPS
#define QMDD_GPU_ZERO_EPS 1e-13
#endif

inline GPUInputHeaderCUDA makeHeader(const GPUInput& in) {
    GPUInputHeaderCUDA h{};
    h.kind = static_cast<uint32_t>(in.kind);
    h.root_re = static_cast<double>(in.root_re);
    h.root_im = static_cast<double>(in.root_im);
    h.dim = static_cast<uint32_t>(in.dim);
    h.pad = in.pad ? in.pad : 1u;
    return h;
}

#define CUDA_CHECK(expr) do {                                  \
    cudaError_t _e = (expr);                                   \
    if (_e != cudaSuccess) {                                   \
        throw std::runtime_error(cudaGetErrorString(_e));      \
    }                                                          \
} while (0)

#endif  // __CUDACC__
#endif  // HEIAN_CUH
