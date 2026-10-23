#ifndef HEIAN_H
#define HEIAN_H

// Metal バックエンドのヘッダ。Heian.mm だけが実装し、CPU 側は USE_METAL のときだけ見る。
// CUDA 版は Heian.cuh。公開 API（BatchJob と各 Wrapper）は両者で一致させること。
//   公開 API を変えたら Heian.cuh も直す
//   Metal 固有部分（GPUInputHeader 以下）は CUDA 側に対応物がある: Heian.cuh の GPUInputHeaderCUDA
//   さらに GPUInputHeader / NormValHost はシェーダ側（Heian.metal）と同じ並びでなければならない
#include <cstddef>
#include <cstdint>
#include <vector>
#include "backend.hpp"

// === 公開 API（Heian.cuh と一致させること）==================================
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

// === 以下は Metal 固有。Heian.mm（Objective-C++）でだけ使う ==================
// CPU 側（通常の g++）もこのヘッダを include するが、ここは素の C++ なのでそのまま通る。
// Metal 固有の API（MTLDevice 等）はこのヘッダには出さない。
#if defined(USE_METAL)

constexpr uint32_t KIND_QMDD     = 1; // SonKind::QMDDNode
constexpr uint32_t KIND_SV       = 2; // SonKind::ArrayLeaf

// Heian.cuh の GPUInputHeaderCUDA と役割が同じ（精度と全ゼロの伝え方だけ違う）:
//   CUDA  … ハンドルをそのまま nullptr で渡し、カーネル内で null 判定
//   Metal … hasRe / hasIm フラグで伝える（nil バッファの参照を避けるため）
// ★ Heian.metal の GPUInputHeader と並び・型を一致させること
struct GPUInputHeader {
    uint32_t kind;
    float    root_re;
    float    root_im;
    uint32_t dim;      // 元の（小さい）一辺
    uint32_t hasRe;    // 0 なら実部バッファ無し（全要素ゼロ）
    uint32_t hasIm;    // 0 なら虚部バッファ無し（全要素ゼロ）
    uint32_t pad;      // 単位行列パディング倍率。実効の一辺は dim*pad
};

// 実効の一辺（パディング後）。Heian.cuh の effDimH と同じ役割
static inline uint32_t effDimH(const GPUInputHeader& h) { return h.dim * h.pad; }

// ★ Heian.metal の NormVal と一致させること
struct NormValHost {
    float    re;
    float    im;
    uint32_t zeroRe;
    uint32_t zeroIm;
};

// Heian.cuh の makeHeader と同じ役割
static inline GPUInputHeader makeHeader(const GPUInput& in) {
    GPUInputHeader h{};
    h.kind = static_cast<uint32_t>(in.kind);
    h.root_re = static_cast<float>(in.root_re);
    h.root_im = static_cast<float>(in.root_im);
    h.dim = static_cast<uint32_t>(in.dim);
    // ArrayLeaf でハンドルが無い成分は全ゼロとして扱う
    h.hasRe = (in.kind == SonKind::ArrayLeaf && in.sv.reHandle == nullptr) ? 0u : 1u;
    h.hasIm = (in.kind == SonKind::ArrayLeaf && in.sv.imHandle == nullptr) ? 0u : 1u;
    h.pad = in.pad ? in.pad : 1u;
    return h;
}

#endif  // USE_METAL
#endif  // HEIAN_H
