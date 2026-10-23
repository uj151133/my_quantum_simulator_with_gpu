#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <mutex>
#include <unordered_map>
#include <map>
#include <tuple>

#include "Heian.cuh"
#include "backend.hpp"
#include "../models/arrayLeaf.hpp"
#include <boost/fiber/operations.hpp>   // GPU 待ちで他の fiber に譲る



// 全要素ゼロの成分（実部 / 虚部）はバッファを保持しない。
// Metal 側（Heian.mm / Heian.metal）にも同じ機能がある。判定の伝え方だけ違う:
//   CUDA  … ハンドルをそのまま nullptr で渡し、カーネル内で null 判定
//   Metal … GPUInputHeader.hasRe / hasIm フラグで伝える（nil バッファの参照を避けるため）

// === GPU 出力バッファのプール（サイズ別フリーリスト） ===========================
// 毎op の cudaMalloc/cudaFree をやめ、解放されたバッファを使い回す。
// ArrayLeaf の releaseGpuBuffer はここに返す（cudaFree しない）→ メモリ churn を止める。
static std::mutex g_poolMx;
static std::unordered_map<size_t, std::vector<void*>> g_freePool;  // bytes → 空きポインタ
static std::unordered_map<void*, size_t> g_ptrSize;                // ポインタ → bytes

// プールが抱えている空きバッファを全て cudaFree して手放す（g_poolMx 保持前提）。
// dim は深さごとに変わるためバケットはサイズ別に複数でき、あるサイズの空きは
// 別サイズの確保には使えない。空きがあるのに OOM するのを避けるための最後の手段。
static size_t poolReclaimLocked() {
    size_t freed = 0;
    for (auto& kv : g_freePool) {
        for (void* q : kv.second) {
            g_ptrSize.erase(q);
            if (cudaFree(q) == cudaSuccess) freed += kv.first;
        }
        kv.second.clear();
    }
    return freed;
}


static void* poolAlloc(size_t bytes) {
    std::lock_guard<std::mutex> lk(g_poolMx);
    auto& bucket = g_freePool[bytes];
    void* p = nullptr;
    if (!bucket.empty()) {
        p = bucket.back();
        bucket.pop_back();
    } else if (cudaMalloc(&p, bytes) != cudaSuccess) {
        // 別サイズのバケットに死蔵された分を返してから確保し直す。
        cudaGetLastError();                      // OOM のエラー状態を消す
        size_t freed = poolReclaimLocked();
        if (freed == 0 || cudaMalloc(&p, bytes) != cudaSuccess) {
            cudaGetLastError();
            size_t freeB = 0, totalB = 0;
            cudaMemGetInfo(&freeB, &totalB);
            throw std::runtime_error(
                "GPU out of memory: " + std::to_string(bytes) + " bytes required, " +
                std::to_string(freeB) + " / " + std::to_string(totalB) + " free"
                " (閾値 PARAMETER.parallelism.GPU を上げてください)");
        }
    }
    g_ptrSize[p] = bytes;
    return p;
}

static void poolFree(void* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_poolMx);
    auto it = g_ptrSize.find(p);
    if (it == g_ptrSize.end()) { cudaFree(p); return; }  // プール外（保険）
    g_freePool[it->second].push_back(p);                  // バケットに返す（cudaFreeしない）
}

namespace {

__device__ inline Cx cmul(Cx a, Cx b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

__device__ inline Cx cadd(Cx a, Cx b) {
    return {a.re + b.re, a.im + b.im};
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
        if (e.childIndex == -1) {
            // 残り (levels-1-level) ビットは identity 拡張: 対角のみ acc、非対角は 0
            uint32_t remaining = levels - 1 - level;
            if (remaining > 0u) {
                uint32_t mask = (1u << remaining) - 1u;
                if ((row & mask) != (col & mask)) {
                    return {0.0, 0.0};
                }
            }
            break;
        }
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
    uint32_t tid,
    bool applyRoot
) {
    // 単位行列パディング（A ⊗ I_pad）を実体化せず添字変換で表す。
    // 非対角ブロックは 0、対角ブロックは元の行列をそのまま引く。
    if (hdr.pad > 1u) {
        if ((row % hdr.pad) != (col % hdr.pad)) return {0.0, 0.0};
        row /= hdr.pad;
        col /= hdr.pad;
        tid = row * hdr.dim + col;
    }
    if (hdr.kind == KIND_SV) {
        // 全要素ゼロの成分はバッファを持たない（容量削減）。その場合は 0 とみなす。
        Cx v{inRe ? inRe[tid] : 0.0, inIm ? inIm[tid] : 0.0};
        return applyRoot ? cmul({hdr.root_re, hdr.root_im}, v) : v;
    } else if (hdr.kind == KIND_QMDD) {
        Cx root = applyRoot ? Cx{hdr.root_re, hdr.root_im} : Cx{1.0, 0.0};
        return evalDD(edges, root, row, col, hdr.dim);
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
    uint32_t dim = effDimH(hdrB);
    uint32_t total = dim * dim;
    if (tid >= total) return;

    uint32_t row = tid / dim;
    uint32_t col = tid - row * dim;

    Cx acc{0.0, 0.0};
    for (uint32_t k = 0; k < dim; ++k) {
        Cx a = evalInput(hdrA, edgesA, inReA, inImA, row, k, row * dim + k, false);
        Cx b = evalInput(hdrB, edgesB, inReB, inImB, k, col, k * dim + col, false);
        acc = cadd(acc, cmul(a, b));
    }

    outRe[tid] = acc.re;
    outIm[tid] = acc.im;
}

// QMDD/SV を dense な dim×dim (re/im) に「一度だけ」展開する。
// QMDD は各マス evalDD を1回（従来 mul 本体で行ごとに dim 回 再走査していたのを排除）。
// SV は既に dense なのでそのままコピー。
__global__ void materialize_kernel(
    GPUInputHeaderCUDA hdr,
    const GPUEdge* edges,
    const double* inRe,
    const double* inIm,
    double* outRe,
    double* outIm,
    uint32_t dim
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = dim * dim;
    if (tid >= total) return;
    uint32_t row = tid / dim;
    uint32_t col = tid - row * dim;

    Cx v{0.0, 0.0};
    if (hdr.kind == KIND_SV) {
        v = Cx{inRe ? inRe[tid] : 0.0, inIm ? inIm[tid] : 0.0};
    } else if (hdr.kind == KIND_QMDD) {
        v = evalDD(edges, Cx{1.0, 0.0}, row, col, dim);
    }
    outRe[tid] = v.re;
    outIm[tid] = v.im;
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
    uint32_t dim = effDimH(hdrB);           // pad 込みの一辺（ラッパーの total と揃える）
    uint32_t total = dim * dim;
    if (tid >= total) return;

    uint32_t row = tid / dim;
    uint32_t col = tid - row * dim;

    Cx a = evalInput(hdrA, edgesA, inReA, inImA, row, col, tid, true);
    Cx b = evalInput(hdrB, edgesB, inReB, inImB, row, col, tid, true);
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
    uint32_t dimA = effDimH(hdrA);
    uint32_t dimB = effDimH(hdrB);
    uint32_t dimOut = dimA * dimB;
    uint32_t total = dimOut * dimOut;
    if (tid >= total) return;

    uint32_t row = tid / dimOut;
    uint32_t col = tid - row * dimOut;

    uint32_t rowA = row / dimB;
    uint32_t colA = col / dimB;
    uint32_t rowB = row % dimB;
    uint32_t colB = col % dimB;

    Cx a = evalInput(hdrA, edgesA, inReA, inImA, rowA, colA, rowA * dimA + colA, false);
    Cx b = evalInput(hdrB, edgesB, inReB, inImB, rowB, colB, rowB * dimB + colB, false);
    Cx v = cmul(a, b);

    outRe[tid] = v.re;
    outIm[tid] = v.im;
}

// === 正規化：L2（Frobenius）ノルム =========================================
// 「最初の非ゼロ要素で割る」方式は割る数の大きさを選ばないため、丸め誤差が
// 係数に選ばれると正規化後の値が際限なく大きくなる（Shor-18 で 10^47 に到達）。
// L2 ノルムは必ず最大要素以上なので、正規化後は定義上 1 以下に収まる。

// ブロックごとの |v|^2 部分和
__global__ void sumsq_partial_kernel(
    const double* re,
    const double* im,
    uint32_t total,
    double* partial
) {
    extern __shared__ double sdata[];
    const uint32_t tid = threadIdx.x;
    const uint32_t i   = blockIdx.x * blockDim.x + threadIdx.x;

    double v = 0.0;
    if (i < total) v = re[i] * re[i] + im[i] * im[i];
    sdata[tid] = v;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// 部分和を集約して coef = sqrt(sum) を書く（全要素ゼロなら coef = 0）
// 集約は 1 スレッドでよい（blocks は grid 数ぶんしかない）
__global__ void norm_finalize_kernel(
    const double* normPartial,
    uint32_t      blocks,
    PostOut*      out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double s = 0.0;
    for (uint32_t i = 0; i < blocks; ++i) s += normPartial[i];
    out->coefRe = (s > 0.0) ? sqrt(s) : 0.0;
    out->coefIm = 0.0;
    // 成分ごとのゼロ判定は「正規化後」の値で行う（eps 未満の丸め込みを反映させるため）。
    // ここでは「ゼロ」を仮置きし、hash_pass1_kernel が非ゼロを見つけたら下ろす。
    out->zeroRe = 1u;
    out->zeroIm = 1u;
}

__global__ void hash_finalize_kernel(
    const uint64_t* hashPartial,
    uint32_t        blocks,
    PostOut*        out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t h = 0;
    for (uint32_t i = 0; i < blocks; ++i) h += hashPartial[i];   // ブロック部分和を加算合成
    out->hash = h;
}

// === 後処理の融合カーネル（1 スレッドブロック） =============================
// ノルム計算 → 正規化 → ハッシュ を __syncthreads() で区切って 1 カーネルで行う。
// 今の使われ方はブロックが 4〜64 要素と非常に小さく、カーネル起動と同期の
// オーバーヘッドが支配的なので、起動を 5 回から 1 回に減らす効果が大きい。
// grid-stride ループなので total に上限は無いが、1 ブロック = 1 SM しか使わない
// ため、大きい total では下の多カーネル版のほうが速い（QMDD_FUSE_MAX で切替）。
__global__ void postprocess_fused_kernel(
    double*  re,
    double*  im,
    uint32_t total,
    PostOut* out
) {
    extern __shared__ char smem[];
    double*   sd = reinterpret_cast<double*>(smem);
    uint64_t* sh = reinterpret_cast<uint64_t*>(smem + blockDim.x * sizeof(double));

    const uint32_t t = threadIdx.x;

    // --- phase 1: |v|^2 の総和 -------------------------------------------
    double acc = 0.0;
    for (uint32_t i = t; i < total; i += blockDim.x) acc += re[i] * re[i] + im[i] * im[i];
    sd[t] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) sd[t] += sd[t + s];
        __syncthreads();
    }
    const double nrm = (sd[0] > 0.0) ? sqrt(sd[0]) : 0.0;
    __syncthreads();

    // --- phase 2: 正規化（と微小値の丸め） --------------------------------
    if (nrm != 0.0) {
        for (uint32_t i = t; i < total; i += blockDim.x) {
            double r = re[i] / nrm;
            double m = im[i] / nrm;
            if (fabs(r) < QMDD_GPU_ZERO_EPS) r = 0.0;
            if (fabs(m) < QMDD_GPU_ZERO_EPS) m = 0.0;
            re[i] = r;
            im[i] = m;
        }
    }
    __syncthreads();

    // --- phase 3: 正規化「後」の値でハッシュ ------------------------------
    // 定数倍だけ違うブロックが同じキーになる必要があるので、必ず正規化後に取る。
    if (t == 0) { out->zeroRe = 1u; out->zeroIm = 1u; }   // 仮置き
    __syncthreads();
    uint64_t h = 0;
    for (uint32_t i = t; i < total; i += blockDim.x) {
        if (re[i] != 0.0) out->zeroRe = 0u;   // 書く値は必ず 0 なので競合しても同じ
        if (im[i] != 0.0) out->zeroIm = 0u;
        uint64_t x = 1469598103934665603ull;
        x = fnv1a64_step(x, static_cast<uint64_t>(i));
        x = fnv1a64_step(x, static_cast<uint64_t>(__double_as_longlong(re[i])));
        x = fnv1a64_step(x, static_cast<uint64_t>(__double_as_longlong(im[i])));
        h += x;   // 加算合成（XORの打ち消しを避ける）
    }
    sh[t] = h;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) { sh[t] += sh[t + s]; }   // 加算合成: 順序非依存かつ打ち消しが起きにくい
        __syncthreads();
    }

    if (t == 0) {
        out->coefRe = nrm;
        out->coefIm = 0.0;
        out->hash   = sh[0];
    }
}

// L2 正規化後は全要素が 1 以下なので、絶対値の閾値がそのまま相対誤差になる。
// 打ち消しの丸め残骸を 0 に丸めておくと、同じ行列が同じビット列になって
// uniqueTable / operationCache が効くようになる（Shor-18 で ops -27%, table -25%）。
// ブロックの Frobenius ノルムは 1 なので、丸めで入る誤差は高々 eps*sqrt(total)。
// coef は実数正なので単純な除算でよい（複素除算の denom アンダーフローが無い）
__global__ void norm_apply_kernel(
    double* re,
    double* im,
    uint32_t total,
    const double* coefRe
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const double c = coefRe[0];
    if (c == 0.0) return;
    double r = re[tid] / c;
    double i = im[tid] / c;
    if (fabs(r) < QMDD_GPU_ZERO_EPS) r = 0.0;
    if (fabs(i) < QMDD_GPU_ZERO_EPS) i = 0.0;
    re[tid] = r;
    im[tid] = i;
}

// === 内容ハッシュ：2 段の並列リダクション ===================================
// 旧実装は <<<1,1>>> の単一スレッドが total 回ループしていた。GPU のコアは
// 単体では非常に遅いので、ブロックを大きくすると線形に効いてくる。
// 要素ごとの FNV をツリーで畳む形にして並列化する。
// 要素ハッシュに位置 (tid) を混ぜているので、並べ替えただけの行列は別の値になる。

__global__ void hash_pass1_kernel(
    const double* re,
    const double* im,
    uint32_t total,
    uint64_t* partial,
    PostOut* out            // 成分ごとのゼロ判定をここで下ろす
) {
    extern __shared__ uint64_t sh[];
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t   = threadIdx.x;

    uint64_t h = 0;
    if (tid < total) {
        // 正規化と eps 丸め込みが済んだ値を見ている。書く値は必ず 0 なので
        // 複数スレッドが同時に書いても結果は同じ（競合しても問題ない）。
        if (re[tid] != 0.0) out->zeroRe = 0u;
        if (im[tid] != 0.0) out->zeroIm = 0u;
        h = 1469598103934665603ull;
        h = fnv1a64_step(h, static_cast<uint64_t>(tid));   // 位置を混ぜる
        h = fnv1a64_step(h, static_cast<uint64_t>(__double_as_longlong(re[tid])));
        h = fnv1a64_step(h, static_cast<uint64_t>(__double_as_longlong(im[tid])));
    }
    sh[t] = h;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            sh[t] += sh[t + s];   // 加算合成（XORの打ち消しを避ける）
        }
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = sh[0];
}



// === 使い回しスクラッチ =======================================================
// cudaMalloc/cudaFree は毎回 GPU と同期する重い処理。固定サイズの一時バッファは
// 一度だけ確保して使い回し、毎op の malloc/free を撲滅する。
// 下の GpuSlot にまとめて持たせ、プールから貸し出す（GpuSlot のコメント参照）。

// run_postprocess 用の固定サイズ一時バッファ（一度確保したら解放しない）
struct PostprocScratch {
    PostOut*      dOut     = nullptr;   // デバイス側の結果 1 個
    PostOut*      hOut     = nullptr;   // ホスト側（pinned）。非同期転送に必須
    double*       dPartial = nullptr;   // ノルムの部分和（grow-only）
    uint64_t*     dHashPartial = nullptr; // ハッシュの部分値（grow-only）
    size_t        partialCap = 0;       // 要素数

    // ストリームは GpuSlot が握る（スロット単位で 1 本）。ここでは結果用バッファだけ用意する。
    void ensureOut() {
        if (dOut) return;               // 初回だけ確保、以降は使い回し
        CUDA_CHECK(cudaMalloc(&dOut, sizeof(PostOut)));
        CUDA_CHECK(cudaHostAlloc(&hOut, sizeof(PostOut), cudaHostAllocDefault));
    }
    void ensurePartial(size_t blocks) { // 足りないときだけ再確保
        if (blocks > partialCap) {
            if (dPartial)     cudaFree(dPartial);
            if (dHashPartial) cudaFree(dHashPartial);
            CUDA_CHECK(cudaMalloc(&dPartial,     blocks * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dHashPartial, blocks * sizeof(uint64_t)));
            partialCap = blocks;
        }
    }
};
// ※ スクラッチ一式は下の GpuSlot にまとめて持たせ、プールから貸し出す。
//    thread_local をやめた理由: GPU 待ちでファイバーを譲るようにすると、
//    ファイバーが別の OS スレッドで再開しうる。そのとき thread_local だと
//    「自分の仕事が載っているのとは別のストリーム」を触ってしまう。

// 可変長 edge 用：必要量が今の容量を超えたときだけ作り直す（grow-only）
struct DeviceEdgeScratch {
    GPUEdge* d   = nullptr;
    size_t   cap = 0;  // 要素数
    GPUEdge* upload(const std::vector<GPUEdge>& edges, cudaStream_t st) {
        if (edges.empty()) return nullptr;
        if (edges.size() > cap) {           // 足りないときだけ再確保
            if (d) cudaFree(d);
            CUDA_CHECK(cudaMalloc(&d, edges.size() * sizeof(GPUEdge)));
            cap = edges.size();
        }
        CUDA_CHECK(cudaMemcpyAsync(d, edges.data(),
                                   edges.size() * sizeof(GPUEdge),
                                   cudaMemcpyHostToDevice, st));
        return d;
    }
};

// dense 化したオペランド用の使い回しバッファ（re/im、grow-only、スレッドごと）
struct DenseScratch {
    double* re = nullptr;
    double* im = nullptr;
    size_t  cap = 0;  // 要素数
    void ensure(size_t n) {
        if (n > cap) {
            if (re) cudaFree(re);
            if (im) cudaFree(im);
            CUDA_CHECK(cudaMalloc(&re, n * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&im, n * sizeof(double)));
            cap = n;
        }
    }
};

// 融合カーネル（1 ブロック）を使う上限。
// 融合版は __syncthreads() で区切る都合上 1 ブロックしか使えないため、total が
// 大きいと 1 SM に全部載って極端に遅くなる。
// RTX 4090 / Shor18 / GPU閾値 12 での実測:
//     多カーネル版 (FUSE_MAX=256)    1211 秒
//     融合版       (FUSE_MAX=65536)  2105 秒   ← 1.74 倍遅い
// 融合版が明確に勝つ領域が無かったので、既定はブロック 1 個分だけに抑える。

// 演算カーネル（mul/add/kron）の 1 ブロックあたりスレッド数。
// total = dim*dim は固定なので、この値を下げるとブロック数が増えて使う SM が増える。
#ifndef QMDD_BLOCK
#define QMDD_BLOCK 256
#endif

#ifndef QMDD_FUSE_MAX
#define QMDD_FUSE_MAX 256u
#endif


} // namespace

// dense 化できる上限。カーネルは tid を uint32_t で索引しているので 2^32 要素は
// そもそも扱えず、8B/要素なのでメモリにも載らない。黙って uint32 をラップさせると
// bytes=0 で確保して未初期化メモリを読むため、ここで明示的に落とす。
static inline uint32_t checkedTotal(const char* op, uint64_t dim) {
    const uint64_t total = dim * dim;
    if (total > 0x7FFFFFFFull) {
        fprintf(stderr,
                "FATAL @%s: dense block too large: dim=%llu -> %llu elements "
                "(%.1f GiB). GPU の dense 化はここまで到達しない前提。"
                "config.yaml の parallelism.GPU を上げて GPU 起動を浅くしてください。\n",
                op, (unsigned long long)dim, (unsigned long long)total,
                (double)total * sizeof(double) * 2 / (1024.0*1024.0*1024.0));
        fflush(stderr);
        abort();
    }
    return static_cast<uint32_t>(total);
}



// ===========================================================================
// === バッチ実行 =============================================================
// 互いに独立な演算をまとめて 1 起動・1 同期で処理する。
// サイズが揃っている必要は無い。「ブロック → 演算番号」の対応表で吸収するので
// パディングは一切しない（小さいジョブを大きく引き伸ばすと計算量が跳ねるため）。
// ===========================================================================

namespace {

constexpr uint32_t BATCH_OP_MUL  = 0;
constexpr uint32_t BATCH_OP_ADD  = 1;
constexpr uint32_t BATCH_OP_KRON = 2;

struct JobDescCUDA {
    GPUInputHeaderCUDA hdrA, hdrB;
    const GPUEdge* edgesA;
    const GPUEdge* edgesB;
    const double*  inReA;  const double* inImA;
    const double*  inReB;  const double* inImB;
    double*        outRe;  double*       outIm;
    uint32_t dim;          // 結果の一辺
    uint32_t total;        // dim * dim
    uint32_t firstBlock;   // このジョブの先頭ブロック番号
    uint32_t partialOff;   // 部分和バッファ内のオフセット
    uint32_t nBlocks;
};

// バッチ用の使い回しバッファ（スロットごと・grow-only）
// グラフにはここの番地が焼き込まれるので、どれかを確保し直したらグラフも捨てる。
struct BatchScratch {
    JobDescCUDA* dDesc      = nullptr;  size_t descCap    = 0;
    uint32_t*    dBlockToOp = nullptr;  size_t blkCap     = 0;
    double*      dPartial   = nullptr;                        // ノルム部分和
    uint64_t*    dHashPart  = nullptr;  size_t partialCap = 0;
    PostOut*     dOut       = nullptr;
    PostOut*     hOut       = nullptr;  size_t outCap     = 0;
    GPUEdge*     dEdges     = nullptr;  size_t edgeCap    = 0; // 全ジョブのエッジを連結

    // --- CUDA Graph 用 -----------------------------------------------------
    // グラフに焼き込む転送元は毎回同じ番地でなければならないので、
    // ホスト側の staging を pinned で固定して持つ（std::vector は番地が動く）。
    JobDescCUDA* hDesc       = nullptr;  size_t hDescCap   = 0;
    uint32_t*    hBlockToOp  = nullptr;  size_t hBlkCap    = 0;
    // (gridバケット, jobバケット, opKind) → 実行可能グラフ
    std::map<std::tuple<uint32_t,uint32_t,uint32_t>, cudaGraphExec_t> graphs;

    void ensureHost(size_t nDesc, size_t nBlk) {
        if (nDesc > hDescCap) {
            if (hDesc) cudaFreeHost(hDesc);
            CUDA_CHECK(cudaHostAlloc(&hDesc, nDesc * sizeof(JobDescCUDA), cudaHostAllocDefault));
            hDescCap = nDesc;
            dropGraphs();                 // 番地が変わったのでグラフは作り直し
        }
        if (nBlk > hBlkCap) {
            if (hBlockToOp) cudaFreeHost(hBlockToOp);
            CUDA_CHECK(cudaHostAlloc(&hBlockToOp, nBlk * sizeof(uint32_t), cudaHostAllocDefault));
            hBlkCap = nBlk;
            dropGraphs();
        }
    }
    void dropGraphs() {
        for (auto& kv : graphs) cudaGraphExecDestroy(kv.second);
        graphs.clear();
    }

    void ensureDesc(size_t n) {
        if (n <= descCap) return;
        if (dDesc) cudaFree(dDesc);
        CUDA_CHECK(cudaMalloc(&dDesc, n * sizeof(JobDescCUDA)));
        descCap = n;
        dropGraphs();                     // 番地が変わったのでグラフは作り直し
    }
    void ensureBlocks(size_t n) {
        if (n <= blkCap) return;
        if (dBlockToOp) cudaFree(dBlockToOp);
        CUDA_CHECK(cudaMalloc(&dBlockToOp, n * sizeof(uint32_t)));
        blkCap = n;
        dropGraphs();
    }
    void ensurePartial(size_t n) {
        if (n <= partialCap) return;
        if (dPartial)  cudaFree(dPartial);
        if (dHashPart) cudaFree(dHashPart);
        CUDA_CHECK(cudaMalloc(&dPartial,  n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dHashPart, n * sizeof(uint64_t)));
        partialCap = n;
        dropGraphs();
    }
    void ensureOut(size_t n) {
        if (n <= outCap) return;
        if (dOut) cudaFree(dOut);
        if (hOut) cudaFreeHost(hOut);
        CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(PostOut)));
        CUDA_CHECK(cudaHostAlloc(&hOut, n * sizeof(PostOut), cudaHostAllocDefault));
        outCap = n;
        dropGraphs();
    }
    void ensureEdges(size_t n) {
        if (n <= edgeCap) return;
        if (dEdges) cudaFree(dEdges);
        CUDA_CHECK(cudaMalloc(&dEdges, n * sizeof(GPUEdge)));
        edgeCap = n;
    }
};

// === GPU 作業スロット =======================================================
// 1 演算ぶんの作業領域一式（ストリーム・イベント・各種スクラッチ）。
// GPU 待ちの間もファイバーが占有し続けるので、プールから貸し出して
// 同時実行数に上限を設ける。これがそのまま背圧になる。
//
// thread_local をやめた理由:
//   待ちでファイバーを譲るようにすると、ファイバーは別の OS スレッドで
//   再開しうる。thread_local のままだと、自分の仕事が載っているのとは
//   別のストリーム・別のスクラッチを触ってしまう。
struct GpuSlot {
    cudaStream_t      stream = nullptr;
    cudaEvent_t       done   = nullptr;   // 完了検知用
    PostprocScratch   post;
    DeviceEdgeScratch edgeA, edgeB;
    DenseScratch      denseA, denseB;
    BatchScratch      batch;

    void ensure() {
        if (stream) return;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        post.ensureOut();
    }
};

// 同時に飛ばせる GPU 演算の上限。環境変数 QMDD_GPU_SLOTS で変えられる。
static int gpuSlotCap() {
    static int cap = [] {
        const char* e = getenv("QMDD_GPU_SLOTS");
        int v = e ? atoi(e) : 64;
        return v > 0 ? v : 64;
    }();
    return cap;
}

static std::mutex           g_slotMx;
static std::vector<GpuSlot*> g_freeSlots;
static int                   g_slotsMade = 0;

static GpuSlot* acquireSlot() {
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(g_slotMx);
            if (!g_freeSlots.empty()) {
                GpuSlot* s = g_freeSlots.back();
                g_freeSlots.pop_back();
                return s;
            }
            if (g_slotsMade < gpuSlotCap()) {
                ++g_slotsMade;
                GpuSlot* s = new GpuSlot();
                s->ensure();
                return s;
            }
        }
        // 空きが無い＝上限まで飛んでいる。他のファイバーに譲って待つ。
        boost::this_fiber::yield();
    }
}

static void releaseSlot(GpuSlot* s) {
    std::lock_guard<std::mutex> lk(g_slotMx);
    g_freeSlots.push_back(s);
}

// 借りたスロットを必ず返すための番人
struct SlotGuard {
    GpuSlot* s;
    SlotGuard() : s(acquireSlot()) {}
    ~SlotGuard() { releaseSlot(s); }
    SlotGuard(const SlotGuard&) = delete;
    SlotGuard& operator=(const SlotGuard&) = delete;
    GpuSlot* operator->() const { return s; }
};

// ストリームの完了を待つ。OS スレッドを止めず、他のファイバーへ譲りながら待つ。
// これをしないと、待っている間その OS スレッドに載っている他のファイバーも
// 道連れで止まり、GPU に投げられる仕事が増えない。
static void waitStream(GpuSlot* s) {
    CUDA_CHECK(cudaEventRecord(s->done, s->stream));
    for (;;) {
        cudaError_t st = cudaEventQuery(s->done);
        if (st == cudaSuccess) return;
        if (st != cudaErrorNotReady) { CUDA_CHECK(st); return; }
        boost::this_fiber::yield();
    }
}



inline void run_postprocess(
    GpuSlot* slot,
    double* dOutRe,
    double* dOutIm,
    uint32_t total,
    int64_t* outId,
    double* outCoef,
    bool* outReIsZero = nullptr,
    bool* outImIsZero = nullptr
) {
    PostprocScratch& g_post = slot->post;
    cudaStream_t st = slot->stream;
    const int block = 256;

    if (total <= QMDD_FUSE_MAX) {
        // --- 融合版：起動 1 回 ------------------------------------------
        const size_t shmem = static_cast<size_t>(block) * (sizeof(double) + sizeof(uint64_t));
        postprocess_fused_kernel<<<1, block, shmem, st>>>(dOutRe, dOutIm, total, g_post.dOut);
    } else {
        // --- 多カーネル版：SM を使い切るためブロックを分ける --------------
        int grid = static_cast<int>((total + block - 1) / block);
        if (grid < 1) grid = 1;
        g_post.ensurePartial(static_cast<size_t>(grid));

        const size_t shmemD = static_cast<size_t>(block) * sizeof(double);
        const size_t shmemU = static_cast<size_t>(block) * sizeof(uint64_t);

        sumsq_partial_kernel<<<grid, block, shmemD, st>>>(dOutRe, dOutIm, total, g_post.dPartial);
        norm_finalize_kernel<<<1, 1, 0, st>>>(g_post.dPartial, static_cast<uint32_t>(grid), g_post.dOut);
        norm_apply_kernel   <<<grid, block, 0, st>>>(dOutRe, dOutIm, total, &g_post.dOut->coefRe);
        // ハッシュは必ず正規化「後」に取る（定数倍違いを同一キーにするため）
        hash_pass1_kernel   <<<grid, block, shmemU, st>>>(dOutRe, dOutIm, total,
                                                          g_post.dHashPartial, g_post.dOut);
        hash_finalize_kernel<<<1, 1, 0, st>>>(g_post.dHashPartial, static_cast<uint32_t>(grid), g_post.dOut);
    }

    // 旧: cudaDeviceSynchronize + cudaMemcpy x3（同期点 4 個）
    // 新: cudaMemcpyAsync x1 + cudaStreamSynchronize x1（同期点 1 個）
    CUDA_CHECK(cudaMemcpyAsync(g_post.hOut, g_post.dOut, sizeof(PostOut),
                               cudaMemcpyDeviceToHost, st));
    waitStream(slot);                       // OS スレッドを止めずに待つ
    outCoef[0] = g_post.hOut->coefRe;
    outCoef[1] = g_post.hOut->coefIm;
    *outId     = static_cast<int64_t>(g_post.hOut->hash);
    if (outReIsZero) *outReIsZero = (g_post.hOut->zeroRe != 0);
    if (outImIsZero) *outImIsZero = (g_post.hOut->zeroIm != 0);
    // ★ ここで cudaFree しない（次の op で使い回す）
}

} // namespace

// --- 演算カーネル（バッチ版）------------------------------------------------
// 中身は既存の *_any2_kernel と同一。添字に [op] が付いただけ。
__global__ void batch_op_kernel(const JobDescCUDA* jobs,
                                const uint32_t* blockToOp,
                                uint32_t opKind) {
    const uint32_t op = blockToOp[blockIdx.x];
    const JobDescCUDA j = jobs[op];
    const uint32_t tid = (blockIdx.x - j.firstBlock) * blockDim.x + threadIdx.x;
    if (tid >= j.total) return;

    Cx v{0.0, 0.0};
    if (opKind == BATCH_OP_MUL) {
        const uint32_t dim = j.dim;
        const uint32_t row = tid / dim, col = tid - row * dim;
        Cx acc{0.0, 0.0};
        for (uint32_t k = 0; k < dim; ++k) {
            Cx a = evalInput(j.hdrA, j.edgesA, j.inReA, j.inImA, row, k, row * dim + k, false);
            Cx b = evalInput(j.hdrB, j.edgesB, j.inReB, j.inImB, k, col, k * dim + col, false);
            acc = cadd(acc, cmul(a, b));
        }
        v = acc;
    } else if (opKind == BATCH_OP_ADD) {
        const uint32_t dim = j.dim;
        const uint32_t row = tid / dim, col = tid - row * dim;
        Cx a = evalInput(j.hdrA, j.edgesA, j.inReA, j.inImA, row, col, tid, true);
        Cx b = evalInput(j.hdrB, j.edgesB, j.inReB, j.inImB, row, col, tid, true);
        v = cadd(a, b);
    } else {  // KRON
        // 実効次元（dim × pad）を使うこと。素の dim で割ると、桁合わせが入った
        // オペランドで行・列の対応が崩れる。単発版 kron_any2_kernel は effDimH を
        // 使っており、バッチ版だけ素の dim なのは取りこぼしだった。
        const uint32_t dimA = effDimH(j.hdrA), dimB = effDimH(j.hdrB);
        const uint32_t dimOut = j.dim;
        const uint32_t row = tid / dimOut, col = tid - row * dimOut;
        const uint32_t rowA = row / dimB, colA = col / dimB;
        const uint32_t rowB = row % dimB, colB = col % dimB;
        Cx a = evalInput(j.hdrA, j.edgesA, j.inReA, j.inImA, rowA, colA, rowA * dimA + colA, false);
        Cx b = evalInput(j.hdrB, j.edgesB, j.inReB, j.inImB, rowB, colB, rowB * dimB + colB, false);
        v = cmul(a, b);
    }
    j.outRe[tid] = v.re;
    j.outIm[tid] = v.im;
}

// --- 後処理（バッチ版）------------------------------------------------------
__global__ void batch_sumsq_kernel(const JobDescCUDA* jobs, const uint32_t* blockToOp,
                                   double* partial) {
    const uint32_t op = blockToOp[blockIdx.x];
    const JobDescCUDA j = jobs[op];
    const uint32_t local = blockIdx.x - j.firstBlock;
    const uint32_t tid   = local * blockDim.x + threadIdx.x;

    extern __shared__ double sdata[];
    const uint32_t t = threadIdx.x;
    double val = 0.0;
    if (tid < j.total) val = j.outRe[tid] * j.outRe[tid] + j.outIm[tid] * j.outIm[tid];
    sdata[t] = val;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) sdata[t] += sdata[t + s];
        __syncthreads();
    }
    // local < j.nBlocks を確認してから書く。パディング用の空ジョブ（nBlocks=0）が
    // 他ジョブの部分和領域へ書き込むのを防ぐ（CUDA Graph 用の詰め物対策）。
    if (t == 0 && local < j.nBlocks) partial[j.partialOff + local] = sdata[0];
}

__global__ void batch_norm_finalize_kernel(const JobDescCUDA* jobs, uint32_t nJobs,
                                           const double* partial, PostOut* out) {
    const uint32_t op = blockIdx.x * blockDim.x + threadIdx.x;
    if (op >= nJobs) return;
    const JobDescCUDA j = jobs[op];
    double s = 0.0;
    for (uint32_t i = 0; i < j.nBlocks; ++i) s += partial[j.partialOff + i];
    out[op].coefRe = (s > 0.0) ? sqrt(s) : 0.0;
    out[op].coefIm = 0.0;
    out[op].zeroRe = 1u;      // 正規化後に非ゼロが見つかったら下ろす
    out[op].zeroIm = 1u;
}

__global__ void batch_norm_apply_kernel(const JobDescCUDA* jobs, const uint32_t* blockToOp,
                                        const PostOut* out) {
    const uint32_t op = blockToOp[blockIdx.x];
    const JobDescCUDA j = jobs[op];
    const uint32_t tid = (blockIdx.x - j.firstBlock) * blockDim.x + threadIdx.x;
    if (tid >= j.total) return;
    const double c = out[op].coefRe;
    if (c == 0.0) return;
    double r = j.outRe[tid] / c;
    double i = j.outIm[tid] / c;
    if (fabs(r) < QMDD_GPU_ZERO_EPS) r = 0.0;
    if (fabs(i) < QMDD_GPU_ZERO_EPS) i = 0.0;
    j.outRe[tid] = r;
    j.outIm[tid] = i;
}

__global__ void batch_hash_pass1_kernel(const JobDescCUDA* jobs, const uint32_t* blockToOp,
                                        uint64_t* partial, PostOut* out) {
    const uint32_t op = blockToOp[blockIdx.x];
    const JobDescCUDA j = jobs[op];
    const uint32_t local = blockIdx.x - j.firstBlock;
    const uint32_t tid   = local * blockDim.x + threadIdx.x;

    extern __shared__ uint64_t sh[];
    const uint32_t t = threadIdx.x;
    uint64_t h = 0;
    if (tid < j.total) {
        if (j.outRe[tid] != 0.0) out[op].zeroRe = 0u;   // 書く値は必ず 0
        if (j.outIm[tid] != 0.0) out[op].zeroIm = 0u;
        h = 1469598103934665603ull;
        h = fnv1a64_step(h, static_cast<uint64_t>(tid));
        h = fnv1a64_step(h, static_cast<uint64_t>(__double_as_longlong(j.outRe[tid])));
        h = fnv1a64_step(h, static_cast<uint64_t>(__double_as_longlong(j.outIm[tid])));
    }
    sh[t] = h;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) { sh[t] += sh[t + s]; }   // 加算合成: 順序非依存かつ打ち消しが起きにくい
        __syncthreads();
    }
    if (t == 0 && local < j.nBlocks) partial[j.partialOff + local] = sh[0];
}

__global__ void batch_hash_finalize_kernel(const JobDescCUDA* jobs, uint32_t nJobs,
                                           const uint64_t* partial, PostOut* out) {
    const uint32_t op = blockIdx.x * blockDim.x + threadIdx.x;
    if (op >= nJobs) return;
    const JobDescCUDA j = jobs[op];
    uint64_t h = 0;
    for (uint32_t i = 0; i < j.nBlocks; ++i) h += partial[j.partialOff + i];   // ブロック部分和を加算合成
    out[op].hash = h;
}


// --- ホスト側の共通処理 -----------------------------------------------------
static void runBatchCommon(uint32_t opKind, BatchJob* jobs, int nJobs) {
    if (nJobs <= 0) return;
    const int block = QMDD_BLOCK;
    SlotGuard slot;                       // 作業一式を借りる（返却は自動）
    BatchScratch& g_batch = slot->batch;
    cudaStream_t st = slot->stream;

    std::vector<JobDescCUDA> desc(nJobs);
    std::vector<uint32_t>    blockToOp;
    std::vector<GPUEdge>     edgeAll;
    // ジョブごとのエッジ配列の開始位置（-1 は QMDDNode でない＝エッジ不要）
    std::vector<int64_t>     offA(nJobs, -1), offB(nJobs, -1);

    uint32_t partialAcc = 0;
    for (int i = 0; i < nJobs; ++i) {
        const GPUInput& A = *jobs[i].A;
        const GPUInput& B = *jobs[i].B;

        const uint64_t dim64 = (opKind == BATCH_OP_KRON)
                             ? static_cast<uint64_t>(effDim(A)) * effDim(B)
                             : static_cast<uint64_t>(effDim(A));
        const uint32_t total = checkedTotal(
            opKind == BATCH_OP_MUL ? "mul(batch)" :
            opKind == BATCH_OP_ADD ? "add(batch)" : "kron(batch)", dim64);
        jobs[i].dim   = static_cast<uint32_t>(dim64);
        jobs[i].outRe = poolAlloc(static_cast<size_t>(total) * sizeof(double));
        jobs[i].outIm = poolAlloc(static_cast<size_t>(total) * sizeof(double));

        JobDescCUDA& d = desc[i];
        d.hdrA = makeHeader(A);
        d.hdrB = makeHeader(B);
        d.inReA = (A.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(A.sv.reHandle) : nullptr;
        d.inImA = (A.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(A.sv.imHandle) : nullptr;
        d.inReB = (B.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(B.sv.reHandle) : nullptr;
        d.inImB = (B.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(B.sv.imHandle) : nullptr;
        if (A.kind == SonKind::QMDDNode && !A.qmdd.edges.empty()) {
            offA[i] = static_cast<int64_t>(edgeAll.size());
            edgeAll.insert(edgeAll.end(), A.qmdd.edges.begin(), A.qmdd.edges.end());
        }
        if (B.kind == SonKind::QMDDNode && !B.qmdd.edges.empty()) {
            offB[i] = static_cast<int64_t>(edgeAll.size());
            edgeAll.insert(edgeAll.end(), B.qmdd.edges.begin(), B.qmdd.edges.end());
        }
        d.outRe = static_cast<double*>(jobs[i].outRe);
        d.outIm = static_cast<double*>(jobs[i].outIm);
        d.dim   = jobs[i].dim;
        d.total = total;

        const uint32_t g = (total + block - 1) / block;
        d.firstBlock = static_cast<uint32_t>(blockToOp.size());
        d.partialOff = partialAcc;
        d.nBlocks    = g;
        blockToOp.insert(blockToOp.end(), g, static_cast<uint32_t>(i));
        partialAcc += g;
    }

    // エッジをまとめて 1 回で転送し、デバイス上のアドレスを解決する
    g_batch.ensureEdges(edgeAll.empty() ? 1 : edgeAll.size());
    if (!edgeAll.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(g_batch.dEdges, edgeAll.data(),
                                   edgeAll.size() * sizeof(GPUEdge),
                                   cudaMemcpyHostToDevice, st));
    }
    for (int i = 0; i < nJobs; ++i) {
        desc[i].edgesA = (offA[i] >= 0) ? (g_batch.dEdges + offA[i]) : nullptr;
        desc[i].edgesB = (offB[i] >= 0) ? (g_batch.dEdges + offB[i]) : nullptr;
    }

    // --- CUDA Graph 用にサイズをバケットへ丸める ---------------------------
    // 6 本のカーネル起動を 1 回のグラフ起動にまとめる（実測 175秒 → 120秒）。
    // グラフを使い回すにはグリッド数とジョブ数が固定でなければならないので、
    // 2 のべき乗に切り上げて、余りは「空のジョブ」で埋める。
    // 空ジョブは total=0 / nBlocks=0 なので、どのカーネルでも即 return する。
    const uint32_t rawGrid = static_cast<uint32_t>(blockToOp.size());
    const uint32_t rawJobs = static_cast<uint32_t>(nJobs);
    auto ceilPow2 = [](uint32_t v) { uint32_t p = 1; while (p < v) p <<= 1; return p; };
    const uint32_t gridTotal = ceilPow2(rawGrid);
    const uint32_t padJobs   = ceilPow2(rawJobs);
    const uint32_t finBlocks = (padJobs + 255u) / 256u;

    g_batch.ensureDesc(padJobs);
    g_batch.ensureBlocks(gridTotal);
    g_batch.ensurePartial(partialAcc ? partialAcc : 1);
    g_batch.ensureOut(padJobs);
    g_batch.ensureHost(padJobs, gridTotal);

    // pinned な固定番地へ詰める（グラフの転送元を毎回同じにするため）
    std::memcpy(g_batch.hDesc, desc.data(), nJobs * sizeof(JobDescCUDA));
    for (uint32_t i = rawJobs; i < padJobs; ++i) {
        std::memset(&g_batch.hDesc[i], 0, sizeof(JobDescCUDA));   // 空ジョブ
    }
    std::memcpy(g_batch.hBlockToOp, blockToOp.data(), rawGrid * sizeof(uint32_t));
    const uint32_t padOp = (padJobs > rawJobs) ? rawJobs : 0u;     // 空ジョブを指す
    for (uint32_t i = rawGrid; i < gridTotal; ++i) g_batch.hBlockToOp[i] = padOp;

    const size_t shmemD = static_cast<size_t>(block) * sizeof(double);
    const size_t shmemU = static_cast<size_t>(block) * sizeof(uint64_t);

    auto encode = [&](cudaStream_t s2) {
        CUDA_CHECK(cudaMemcpyAsync(g_batch.dDesc, g_batch.hDesc,
                                   padJobs * sizeof(JobDescCUDA), cudaMemcpyHostToDevice, s2));
        CUDA_CHECK(cudaMemcpyAsync(g_batch.dBlockToOp, g_batch.hBlockToOp,
                                   gridTotal * sizeof(uint32_t), cudaMemcpyHostToDevice, s2));
        batch_op_kernel           <<<gridTotal, block, 0, s2>>>(g_batch.dDesc, g_batch.dBlockToOp, opKind);
        batch_sumsq_kernel        <<<gridTotal, block, shmemD, s2>>>(g_batch.dDesc, g_batch.dBlockToOp, g_batch.dPartial);
        batch_norm_finalize_kernel<<<finBlocks, 256, 0, s2>>>(g_batch.dDesc, padJobs, g_batch.dPartial, g_batch.dOut);
        batch_norm_apply_kernel   <<<gridTotal, block, 0, s2>>>(g_batch.dDesc, g_batch.dBlockToOp, g_batch.dOut);
        batch_hash_pass1_kernel   <<<gridTotal, block, shmemU, s2>>>(g_batch.dDesc, g_batch.dBlockToOp, g_batch.dHashPart, g_batch.dOut);
        batch_hash_finalize_kernel<<<finBlocks, 256, 0, s2>>>(g_batch.dDesc, padJobs, g_batch.dHashPart, g_batch.dOut);
        CUDA_CHECK(cudaMemcpyAsync(g_batch.hOut, g_batch.dOut,
                                   padJobs * sizeof(PostOut), cudaMemcpyDeviceToHost, s2));
    };

    const auto key = std::make_tuple(gridTotal, padJobs, opKind);
    auto it = g_batch.graphs.find(key);
    if (it == g_batch.graphs.end()) {
        // 同じ形は繰り返し来るので、一度キャプチャして使い回す
        cudaGraph_t g = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        encode(st);
        CUDA_CHECK(cudaStreamEndCapture(st, &g));
        cudaGraphExec_t ge = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(g));
        it = g_batch.graphs.emplace(key, ge).first;
    }
    CUDA_CHECK(cudaGraphLaunch(it->second, st));   // 起動は 1 回だけ

    waitStream(slot.s);                         // ← 同期はここ 1 回だけ（譲りながら待つ）

    for (int i = 0; i < nJobs; ++i) {
        const PostOut& o = g_batch.hOut[i];
        const bool zr = (o.zeroRe != 0), zi = (o.zeroIm != 0);
        if (zr && zi) {                          // 全要素ゼロ → ゼロ行列
            poolFree(jobs[i].outRe); jobs[i].outRe = nullptr;
            poolFree(jobs[i].outIm); jobs[i].outIm = nullptr;
            jobs[i].outCoef[0] = 0; jobs[i].outCoef[1] = 0;
            jobs[i].outId = 0;
            continue;
        }
        if (zr) { poolFree(jobs[i].outRe); jobs[i].outRe = nullptr; }
        if (zi) { poolFree(jobs[i].outIm); jobs[i].outIm = nullptr; }

        double cr = o.coefRe, ci = o.coefIm;
        if (opKind != BATCH_OP_ADD) {            // mul / kron は root 重みを掛ける
            const GPUInput& A = *jobs[i].A;
            const GPUInput& B = *jobs[i].B;
            const double rA = A.root_re, iA = A.root_im;
            const double rB = B.root_re, iB = B.root_im;
            const double r = rA * rB - iA * iB;
            const double m = rA * iB + iA * rB;
            const double nr = cr * r - ci * m;
            const double ni = cr * m + ci * r;
            cr = nr; ci = ni;
        }
        jobs[i].outCoef[0] = static_cast<gpu_precision>(cr);
        jobs[i].outCoef[1] = static_cast<gpu_precision>(ci);
        jobs[i].outId      = static_cast<int64_t>(o.hash);
    }

}

extern "C" void runMulBatchWrapper(BatchJob* jobs, int n)  { runBatchCommon(BATCH_OP_MUL,  jobs, n); }
extern "C" void runAddBatchWrapper(BatchJob* jobs, int n)  { runBatchCommon(BATCH_OP_ADD,  jobs, n); }
extern "C" void runKronBatchWrapper(BatchJob* jobs, int n) { runBatchCommon(BATCH_OP_KRON, jobs, n); }

extern "C" void runMulAny2Wrapper(
    const GPUInput& A, const GPUInput& B,
    void** outRe, void** outIm, int64_t* outId, gpu_precision* outCoef
) {
    SlotGuard slot;                     // 作業一式を借りる（返却は自動）
    GPUInputHeaderCUDA hdrA = makeHeader(A);
    GPUInputHeaderCUDA hdrB = makeHeader(B);

    const uint32_t dim = effDimH(hdrA);
    const uint32_t total = checkedTotal("mul", static_cast<uint64_t>(dim));
    const size_t bytes = static_cast<size_t>(total) * sizeof(double);

    auto* dInReA = (A.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(A.sv.reHandle) : nullptr;
    auto* dInImA = (A.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(A.sv.imHandle) : nullptr;
    auto* dInReB = (B.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(B.sv.reHandle) : nullptr;
    auto* dInImB = (B.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(B.sv.imHandle) : nullptr;
    GPUEdge* dEdgesA = (A.kind == SonKind::QMDDNode) ? slot->edgeA.upload(A.qmdd.edges, slot->stream) : nullptr;
    GPUEdge* dEdgesB = (B.kind == SonKind::QMDDNode) ? slot->edgeB.upload(B.qmdd.edges, slot->stream) : nullptr;

    double* dOutRe = nullptr;
    double* dOutIm = nullptr;
    dOutRe = (double*)poolAlloc(bytes);   // プールから取得（無ければ cudaMalloc）
    dOutIm = (double*)poolAlloc(bytes);
    int block = QMDD_BLOCK;
    int grid = static_cast<int>((total + block - 1) / block);

    // --- Stage 1: QMDD オペランドを dense 化（SV はハンドルをそのまま使う）---
    const double* aRe = dInReA; const double* aIm = dInImA;
    const double* bRe = dInReB; const double* bIm = dInImB;
    if (A.kind == SonKind::QMDDNode) {
        slot->denseA.ensure(total);
        materialize_kernel<<<grid, block, 0, slot->stream>>>(hdrA, dEdgesA, nullptr, nullptr, slot->denseA.re, slot->denseA.im, dim);
        aRe = slot->denseA.re; aIm = slot->denseA.im;
    }
    if (B.kind == SonKind::QMDDNode) {
        slot->denseB.ensure(total);
        materialize_kernel<<<grid, block, 0, slot->stream>>>(hdrB, dEdgesB, nullptr, nullptr, slot->denseB.re, slot->denseB.im, dim);
        bRe = slot->denseB.re; bIm = slot->denseB.im;
    }

    // mul_any2_kernel を「両方 SV 扱い（dense）」で実行：evalDD 無しの素直な dense matmul
    // materialize 済みの dense は既に実効サイズ（パディング適用後）なので、
    // ここで pad を 1 に戻さないと evalInput が二重にパディングを掛けてしまう。
    GPUInputHeaderCUDA svA = hdrA; svA.kind = KIND_SV; svA.dim = dim; svA.pad = 1u;
    GPUInputHeaderCUDA svB = hdrB; svB.kind = KIND_SV; svB.dim = dim; svB.pad = 1u;
    if (A.kind == SonKind::ArrayLeaf) { svA.dim = hdrA.dim; svA.pad = hdrA.pad; }  // 未展開なら元のまま
    if (B.kind == SonKind::ArrayLeaf) { svB.dim = hdrB.dim; svB.pad = hdrB.pad; }
    mul_any2_kernel<<<grid, block, 0, slot->stream>>>(
        svA, svB, dEdgesA, dEdgesB,
        aRe, aIm, bRe, bIm,
        dOutRe, dOutIm
    );

    double coef[2] = {0.0, 0.0};
    bool reIsZero = false, imIsZero = false;
    run_postprocess(slot.s, dOutRe, dOutIm, total, outId, coef, &reIsZero, &imIsZero);
    if (reIsZero && imIsZero) {
        // 正規化後に全要素が 0 に丸められた → ゼロ行列。coef を 0 にして
        // 呼び出し側（mathUtils）に edgeZero を返させる。
        poolFree(dOutRe); dOutRe = nullptr;
        poolFree(dOutIm); dOutIm = nullptr;
        coef[0] = 0.0; coef[1] = 0.0;
    } else {
        if (reIsZero) { poolFree(dOutRe); dOutRe = nullptr; }  // 実部が全ゼロなら保持しない
        if (imIsZero) { poolFree(dOutIm); dOutIm = nullptr; }  // 虚部が全ゼロなら保持しない
    }

    double rA = static_cast<double>(A.root_re), iA = static_cast<double>(A.root_im);
    double rB = static_cast<double>(B.root_re), iB = static_cast<double>(B.root_im);
    double r = rA * rB - iA * iB;
    double i = rA * iB + iA * rB;
    outCoef[0] = coef[0] * r - coef[1] * i;
    outCoef[1] = coef[0] * i + coef[1] * r;

    if (outRe) *outRe = dOutRe;
    if (outIm) *outIm = dOutIm;

    // edges はスロットの edgeA / edgeB で使い回すので cudaFree しない
}

extern "C" void runAddAny2Wrapper(
    const GPUInput& A, const GPUInput& B,
    void** outRe, void** outIm, int64_t* outId, gpu_precision* outCoef
) {
    SlotGuard slot;                     // 作業一式を借りる（返却は自動）
    GPUInputHeaderCUDA hdrA = makeHeader(A);
    GPUInputHeaderCUDA hdrB = makeHeader(B);

    const uint32_t dim = effDimH(hdrA);
    const uint32_t total = checkedTotal("add", static_cast<uint64_t>(dim));
    const size_t bytes = static_cast<size_t>(total) * sizeof(double);

    auto* dInReA = (A.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(A.sv.reHandle) : nullptr;
    auto* dInImA = (A.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(A.sv.imHandle) : nullptr;
    auto* dInReB = (B.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(B.sv.reHandle) : nullptr;
    auto* dInImB = (B.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(B.sv.imHandle) : nullptr;

    GPUEdge* dEdgesA = (A.kind == SonKind::QMDDNode) ? slot->edgeA.upload(A.qmdd.edges, slot->stream) : nullptr;
    GPUEdge* dEdgesB = (B.kind == SonKind::QMDDNode) ? slot->edgeB.upload(B.qmdd.edges, slot->stream) : nullptr;

    double* dOutRe = nullptr;
    double* dOutIm = nullptr;
    dOutRe = (double*)poolAlloc(bytes);   // プールから取得（無ければ cudaMalloc）
    dOutIm = (double*)poolAlloc(bytes);

    int block = QMDD_BLOCK;
    int grid = static_cast<int>((total + block - 1) / block);
    add_any2_kernel<<<grid, block, 0, slot->stream>>>(
        hdrA, hdrB, dEdgesA, dEdgesB,
        dInReA, dInImA, dInReB, dInImB,
        dOutRe, dOutIm
    );

    double coef[2] = {0.0, 0.0};
    bool reIsZero = false, imIsZero = false;
    run_postprocess(slot.s, dOutRe, dOutIm, total, outId, coef, &reIsZero, &imIsZero);
    if (reIsZero && imIsZero) {
        // 正規化後に全要素が 0 に丸められた → ゼロ行列。coef を 0 にして
        // 呼び出し側（mathUtils）に edgeZero を返させる。
        poolFree(dOutRe); dOutRe = nullptr;
        poolFree(dOutIm); dOutIm = nullptr;
        coef[0] = 0.0; coef[1] = 0.0;
    } else {
        if (reIsZero) { poolFree(dOutRe); dOutRe = nullptr; }  // 実部が全ゼロなら保持しない
        if (imIsZero) { poolFree(dOutIm); dOutIm = nullptr; }  // 虚部が全ゼロなら保持しない
    }
    outCoef[0] = coef[0];
    outCoef[1] = coef[1];

    if (outRe) *outRe = dOutRe;
    if (outIm) *outIm = dOutIm;

    // edges はスロットの edgeA / edgeB で使い回すので cudaFree しない
}

extern "C" void runKronAny2Wrapper(
    const GPUInput& A, const GPUInput& B,
    void** outRe, void** outIm, int64_t* outId, gpu_precision* outCoef
) {
    SlotGuard slot;                     // 作業一式を借りる（返却は自動）
    GPUInputHeaderCUDA hdrA = makeHeader(A);
    GPUInputHeaderCUDA hdrB = makeHeader(B);

    const uint64_t dimOut64 = static_cast<uint64_t>(effDimH(hdrA)) * effDimH(hdrB);
    const uint32_t total = checkedTotal("kron", dimOut64);
    const uint32_t dimOut = static_cast<uint32_t>(dimOut64);
    const size_t bytes = static_cast<size_t>(total) * sizeof(double);

    auto* dInReA = (A.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(A.sv.reHandle) : nullptr;
    auto* dInImA = (A.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(A.sv.imHandle) : nullptr;
    auto* dInReB = (B.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(B.sv.reHandle) : nullptr;
    auto* dInImB = (B.kind == SonKind::ArrayLeaf) ? static_cast<const double*>(B.sv.imHandle) : nullptr;

    GPUEdge* dEdgesA = (A.kind == SonKind::QMDDNode) ? slot->edgeA.upload(A.qmdd.edges, slot->stream) : nullptr;
    GPUEdge* dEdgesB = (B.kind == SonKind::QMDDNode) ? slot->edgeB.upload(B.qmdd.edges, slot->stream) : nullptr;

    double* dOutRe = nullptr;
    double* dOutIm = nullptr;
    dOutRe = (double*)poolAlloc(bytes);   // プールから取得（無ければ cudaMalloc）
    dOutIm = (double*)poolAlloc(bytes);

    int block = QMDD_BLOCK;
    int grid = static_cast<int>((total + block - 1) / block);
    kron_any2_kernel<<<grid, block, 0, slot->stream>>>(
        hdrA, hdrB, dEdgesA, dEdgesB,
        dInReA, dInImA, dInReB, dInImB,
        dOutRe, dOutIm
    );

    double coef[2] = {0.0, 0.0};
    bool reIsZero = false, imIsZero = false;
    run_postprocess(slot.s, dOutRe, dOutIm, total, outId, coef, &reIsZero, &imIsZero);
    if (reIsZero && imIsZero) {
        // 正規化後に全要素が 0 に丸められた → ゼロ行列。coef を 0 にして
        // 呼び出し側（mathUtils）に edgeZero を返させる。
        poolFree(dOutRe); dOutRe = nullptr;
        poolFree(dOutIm); dOutIm = nullptr;
        coef[0] = 0.0; coef[1] = 0.0;
    } else {
        if (reIsZero) { poolFree(dOutRe); dOutRe = nullptr; }  // 実部が全ゼロなら保持しない
        if (imIsZero) { poolFree(dOutIm); dOutIm = nullptr; }  // 虚部が全ゼロなら保持しない
    }

    double rA = static_cast<double>(A.root_re), iA = static_cast<double>(A.root_im);
    double rB = static_cast<double>(B.root_re), iB = static_cast<double>(B.root_im);
    double r = rA * rB - iA * iB;
    double i = rA * iB + iA * rB;
    outCoef[0] = coef[0] * r - coef[1] * i;
    outCoef[1] = coef[0] * i + coef[1] * r;

    if (outRe) *outRe = dOutRe;
    if (outIm) *outIm = dOutIm;

    // edges はスロットの edgeA / edgeB で使い回すので cudaFree しない
}

extern "C" void releaseGpuBuffer(void* p) {
    poolFree(p);   // cudaFree せずプールに返して使い回す
}

// 旧 API 互換。CUDA のバッファは double なのでここで有効桁が落ちる。
// ArrayLeaf の読み出しには copyGpuBufferToHostDouble を使うこと。
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




