#include <metal_stdlib>
using namespace metal;

struct GPUEdge {
    float re;
    float im;
    int    childIndex; // -1 = terminal
};

struct GPUInputHeader {
    uint kind;
    float root_re;
    float root_im;
    uint dim;      // sv.size（=行列の一辺）
    uint hasRe;    // 0 なら実部バッファを持たない（全要素ゼロ）。inRe は参照しない
    uint hasIm;    // 0 なら虚部バッファを持たない（全要素ゼロ）。inIm は参照しない
    uint pad;      // 単位行列パディング倍率。実効の一辺は dim*pad。Heian.mm と一致させること
};

// 実効の一辺（パディング後）
inline uint effDim(const device GPUInputHeader& h) { return h.dim * h.pad; }

struct NormVal {
    float re;
    float im;
    uint  zeroRe;   // 実部が全要素ゼロなら 1
    uint  zeroIm;   // 虚部が全要素ゼロなら 1
};

// === 正規化：L2（Frobenius）ノルム =========================================
// 「最初の非ゼロ要素で割る」方式は割る数の大きさを選ばないため、丸め誤差が
// 係数に選ばれると正規化後の値が際限なく大きくなる。float では上限 3.4e38 を
// 容易に超え、複素除算の denom = |c|^2 も |c| < 3.7e-23 でゼロに落ちる。
// L2 ノルムは必ず最大要素以上なので、正規化後は定義上 1 以下に収まる。

// スレッドグループごとの |v|^2 部分和
kernel void norm_pass1(
    const device float* outRe      [[buffer(0)]],
    const device float* outIm      [[buffer(1)]],
    device float* groupSum          [[buffer(2)]],
    constant uint& total            [[buffer(3)]],
    uint tid                        [[thread_position_in_grid]],
    uint tg_id                      [[threadgroup_position_in_grid]],
    uint tg_tid                     [[thread_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    float v = 0.0;
    if (tid < total) {
        float re = outRe[tid];
        float im = outIm[tid];
        v = re * re + im * im;
    }

    threadgroup float sdata[1024];
    sdata[tg_tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tg_tid < s) sdata[tg_tid] += sdata[tg_tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tg_tid == 0) groupSum[tg_id] = sdata[0];
}

// 部分和を集約して coef = sqrt(sum) を書く（全要素ゼロなら coef = 0）
kernel void norm_pass2(
    const device float* groupSum    [[buffer(0)]],
    device NormVal* outCoef         [[buffer(1)]],
    constant uint& groupCount       [[buffer(2)]],
    uint tid                        [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    float s = 0.0;
    for (uint i = 0; i < groupCount; ++i) s += groupSum[i];
    // 成分ごとのゼロ判定は「正規化後」の値で行う（eps 未満の丸め込みを反映させるため）。
    // ここでは「ゼロ」を仮置きし、hash_pass1 が非ゼロを見つけたら下ろす。
    outCoef[0] = NormVal{ (s > 0.0) ? sqrt(s) : 0.0f, 0.0f, 1u, 1u };
}

// L2 正規化後は全要素が 1 以下なので、絶対値の閾値がそのまま相対誤差になる。
// float の機械イプシロンは約 1.2e-7 なので、その少し上を打ち切り閾値にする。
#define QMDD_GPU_ZERO_EPS 1e-6f

// coef は実数正なので単純な除算でよい（複素除算の denom アンダーフローが無い）
kernel void norm_apply(
    device float* outRe             [[buffer(0)]],
    device float* outIm             [[buffer(1)]],
    const device NormVal* coefBuf   [[buffer(2)]],
    constant uint& total            [[buffer(3)]],
    uint tid                        [[thread_position_in_grid]]
) {
    if (tid >= total) return;
    float c = coefBuf[0].re;
    if (c == 0.0) return;
    float r = outRe[tid] / c;
    float i = outIm[tid] / c;
    if (fabs(r) < QMDD_GPU_ZERO_EPS) r = 0.0;
    if (fabs(i) < QMDD_GPU_ZERO_EPS) i = 0.0;
    outRe[tid] = r;
    outIm[tid] = i;
}

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline ulong fnv1a64_step(ulong h, ulong v) {
    h ^= v;
    h *= 1099511628211ul;
    return h;
}

kernel void hash_pass1(
    const device float* outRe [[buffer(0)]],
    const device float* outIm [[buffer(1)]],
    device ulong* partial      [[buffer(2)]],
    constant uint& total                 [[buffer(3)]],
    device NormVal* outCoef    [[buffer(4)]],   // 成分ごとのゼロ判定をここで下ろす
    uint tid                   [[thread_position_in_grid]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint tg_tid                [[thread_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    // 担当が無いスレッドも threadgroup_barrier には到達させる必要がある。
    // 途中で return するとバリアに来ないスレッドが出て未定義動作になり、
    // さらに sh[] の未初期化分が縮約に混ざってハッシュが壊れる。
    // 担当外は h = 0 を置くだけにする（CUDA 側 hash_pass1_kernel と同じ形）。
    ulong h = 0;
    if (tid < total) {
        // 正規化と eps 丸め込みが済んだ値を見ている。書く値は必ず 0 なので
        // 複数スレッドが同時に書いても結果は同じ（競合しても問題ない）。
        if (outRe[tid] != 0.0f) outCoef[0].zeroRe = 0u;
        if (outIm[tid] != 0.0f) outCoef[0].zeroIm = 0u;

        h = 1469598103934665603ul;
        h = fnv1a64_step(h, (ulong)tid);   // 位置を混ぜる（並べ替えを別物にする）
        h = fnv1a64_step(h, (ulong)as_type<uint>(outRe[tid]));
        h = fnv1a64_step(h, (ulong)as_type<uint>(outIm[tid]));
    }

    threadgroup ulong sh[1024];
    sh[tg_tid] = h;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tg_tid < s) sh[tg_tid] += sh[tg_tid + s];   // 加算合成: 順序非依存かつ打ち消しが起きにくい（Heian.cu と同じ）
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tg_tid == 0) {
        partial[tg_id] = sh[0];
    }
}

kernel void hash_pass2(
    const device ulong* partial [[buffer(0)]],
    device ulong* outId          [[buffer(1)]],
    constant uint& partialCount  [[buffer(2)]],
    uint tid                     [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    ulong h = 0;
    for (uint i = 0; i < partialCount; ++i) h += partial[i];   // 部分和を加算合成（Heian.cu と同じ）
    outId[0] = h;
}

inline float2 evalDD(
    const device GPUEdge* edges,
    float2 rootW,
    uint row,
    uint col,
    uint dim
) {
    uint levels = 0;
    uint t = dim;
    while (t > 1) { t >>= 1; ++levels; }

    float2 acc = rootW;
    int node = 0;

    for (uint level = 0; level < levels; ++level) {
        uint shift = (levels - 1 - level);
        uint rb = (row >> shift) & 1u;
        uint cb = (col >> shift) & 1u;
        uint k  = (rb << 1) | cb;
        const GPUEdge e = edges[node * 4 + k];
        acc = cmul(acc, float2(e.re, e.im));
        if (e.childIndex == -1) {
            // 残り (levels-1-level) ビットは identity 拡張: 対角のみ acc、非対角は 0
            uint remaining = levels - 1 - level;
            if (remaining > 0u) {
                uint mask = (1u << remaining) - 1u;
                if ((row & mask) != (col & mask)) {
                    return float2(0.0, 0.0);
                }
            }
            break;
        }
        node = e.childIndex;
    }
    return acc;
}

inline float2 evalInput(
    const device GPUInputHeader& hdr,
    const device GPUEdge* edges,
    const device float* inRe,
    const device float* inIm,
    uint row,
    uint col,
    uint tid,
    bool applyRoot
) {
    // 単位行列パディング（A ⊗ I_pad）を実体化せず添字変換で表す。
    // 非対角ブロックは 0、対角ブロックは元の行列をそのまま引く。
    // Heian.cu の evalInput と同じ規約にすること。
    if (hdr.pad > 1u) {
        if ((row % hdr.pad) != (col % hdr.pad)) return float2(0.0, 0.0);
        row /= hdr.pad;
        col /= hdr.pad;
        tid = row * hdr.dim + col;
    }
    if (hdr.kind == 2) { // SV
        // 全要素ゼロの成分はバッファを持たない（容量削減）。そのポインタは触らない。
        float2 v = float2(hdr.hasRe != 0u ? inRe[tid] : 0.0f,
                          hdr.hasIm != 0u ? inIm[tid] : 0.0f);
        return applyRoot ? cmul(float2(hdr.root_re, hdr.root_im), v) : v;
    } else if (hdr.kind == 1) { // QMDD
        float2 root = applyRoot ? float2(hdr.root_re, hdr.root_im) : float2(1.0, 0.0);
        return evalDD(edges, root, row, col, hdr.dim);
    } else { // Terminal
        return float2(0.0, 0.0);
    }
}

// QMDD/SV を dense な dim×dim (re/im) に「一度だけ」展開する。
// QMDD は各マス evalDD を1回（従来は mul 本体で行ごとに dim 回 再走査していた）。
// SV は既に dense なのでそのままコピー。
kernel void materialize_any(
    const device GPUInputHeader& hdr [[buffer(0)]],
    const device GPUEdge* edges      [[buffer(1)]],
    const device float* inRe         [[buffer(2)]],
    const device float* inIm         [[buffer(3)]],
    device float* outRe              [[buffer(4)]],
    device float* outIm              [[buffer(5)]],
    constant uint& dim               [[buffer(6)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint total = dim * dim;
    if (tid >= total) return;
    uint row = tid / dim;
    uint col = tid - row * dim;

    float2 v;
    if (hdr.kind == 2u) {            // SV: 既に dense
        v = float2(hdr.hasRe != 0u ? inRe[tid] : 0.0f,
                   hdr.hasIm != 0u ? inIm[tid] : 0.0f);
    } else if (hdr.kind == 1u) {     // QMDD: evalDD を1回（applyRoot=false 相当の root=(1,0)）
        v = evalDD(edges, float2(1.0, 0.0), row, col, dim);
    } else {
        v = float2(0.0, 0.0);
    }
    outRe[tid] = v.x;
    outIm[tid] = v.y;
}

kernel void mul_any2(
    const device GPUInputHeader& hdrA [[buffer(0)]],
    const device GPUInputHeader& hdrB [[buffer(1)]],
    const device GPUEdge* edgesA      [[buffer(2)]],
    const device GPUEdge* edgesB      [[buffer(3)]],
    const device float* inReA        [[buffer(4)]],
    const device float* inImA        [[buffer(5)]],
    const device float* inReB        [[buffer(6)]],
    const device float* inImB        [[buffer(7)]],
    device float* outRe              [[buffer(8)]],
    device float* outIm              [[buffer(9)]],
    uint tid                          [[thread_position_in_grid]]
) {
    uint dim = effDim(hdrB);    // 前提：AとBで同じ（パディング後の実効サイズ）
    uint total = dim * dim;
    if (tid >= total) return;

    uint row = tid / dim;
    uint col = tid - row * dim;

    float2 acc = float2(0.0, 0.0);

    for (uint k = 0; k < dim; ++k) {
        // pad == 1 の密葉は渡した tid をそのまま引くので、実位置を渡す必要がある。
        // （以前は 0 を渡していて、pad == 1 の SV では常に要素 0 を読んでいた。Heian.cu と同じ形に修正）
        float2 a = evalInput(hdrA, edgesA, inReA, inImA, row, k, row * dim + k, false);
        float2 b = evalInput(hdrB, edgesB, inReB, inImB, k, col, k * dim + col, false);
        acc += cmul(a, b);
    }

    outRe[tid] = acc.x;
    outIm[tid] = acc.y;
}

kernel void add_any2(
    const device GPUInputHeader& hdrA [[buffer(0)]],
    const device GPUInputHeader& hdrB [[buffer(1)]],
    const device GPUEdge* edgesA      [[buffer(2)]],
    const device GPUEdge* edgesB      [[buffer(3)]],
    const device float* inReA        [[buffer(4)]],
    const device float* inImA        [[buffer(5)]],
    const device float* inReB        [[buffer(6)]],
    const device float* inImB        [[buffer(7)]],
    device float* outRe              [[buffer(8)]],
    device float* outIm              [[buffer(9)]],
    uint tid                          [[thread_position_in_grid]]
) {
    uint dim = effDim(hdrB); // 前提：AとBで同じ（パディング後の実効サイズ）
    uint total = dim * dim;
    if (tid >= total) return;

    uint row = tid / dim;
    uint col = tid - row * dim;

    float2 a = evalInput(hdrA, edgesA, inReA, inImA, row, col, tid, true);
    float2 b = evalInput(hdrB, edgesB, inReB, inImB, row, col, tid, true);
    float2 v = a + b;

    outRe[tid] = v.x;
    outIm[tid] = v.y;
}

kernel void kron_any2(
    const device GPUInputHeader& hdrA [[buffer(0)]],
    const device GPUInputHeader& hdrB [[buffer(1)]],
    const device GPUEdge* edgesA      [[buffer(2)]],
    const device GPUEdge* edgesB      [[buffer(3)]],
    const device float* inReA        [[buffer(4)]],
    const device float* inImA        [[buffer(5)]],
    const device float* inReB        [[buffer(6)]],
    const device float* inImB        [[buffer(7)]],
    device float* outRe              [[buffer(8)]],
    device float* outIm              [[buffer(9)]],
    uint tid                          [[thread_position_in_grid]]
) {
    uint dimA = effDim(hdrA);
    uint dimB = effDim(hdrB);
    uint dimOut = dimA * dimB;
    uint total = dimOut * dimOut;
    if (tid >= total) return;

    uint row = tid / dimOut;
    uint col = tid - row * dimOut;

    uint rowA = row / dimB;
    uint colA = col / dimB;
    uint rowB = row % dimB;
    uint colB = col % dimB;

    float2 a = evalInput(hdrA, edgesA, inReA, inImA, rowA, colA, rowA * dimA + colA, false);
    float2 b = evalInput(hdrB, edgesB, inReB, inImB, rowB, colB, rowB * dimB + colB, false);
    // ↑ pad>1 のときは evalInput 側で row/col から tid を引き直す

    float2 v = cmul(a, b);
    outRe[tid] = v.x;
    outIm[tid] = v.y;
}
// ============================================================================
// 単発の後処理：融合版（ノルム → 正規化 → ハッシュ を 1 カーネル）
// Heian.cu の postprocess_fused_kernel と同じ。total が QMDD_FUSE_MAX 以下のときに使う。
// threadgroup 1 つ・POST_TG スレッドで起動すること（縮約が 2 のべき乗を前提にしている）。
// 結果は多カーネル版と同じく outCoef（NormVal）と outId（ulong）に書く。
// ============================================================================
constant constexpr uint POST_TG = 256;

kernel void postprocess_fused(
    device float*   re      [[buffer(0)]],
    device float*   im      [[buffer(1)]],
    device NormVal* outCoef [[buffer(2)]],
    device ulong*   outId   [[buffer(3)]],
    constant uint&  total   [[buffer(4)]],
    uint t                  [[thread_index_in_threadgroup]],
    uint tgsz               [[threads_per_threadgroup]]
) {
    threadgroup float sd[POST_TG];
    threadgroup ulong sh[POST_TG];

    // --- phase 1: |v|^2 の総和 -------------------------------------------
    float acc = 0.0f;
    for (uint i = t; i < total; i += tgsz) acc += re[i] * re[i] + im[i] * im[i];
    sd[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz / 2; s > 0; s >>= 1) {
        if (t < s) sd[t] += sd[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float nrm = (sd[0] > 0.0f) ? sqrt(sd[0]) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- phase 2: 正規化（と微小値の丸め） --------------------------------
    if (nrm != 0.0f) {
        for (uint i = t; i < total; i += tgsz) {
            float r = re[i] / nrm;
            float m = im[i] / nrm;
            if (fabs(r) < QMDD_GPU_ZERO_EPS) r = 0.0f;
            if (fabs(m) < QMDD_GPU_ZERO_EPS) m = 0.0f;
            re[i] = r;
            im[i] = m;
        }
    }
    // phase 2 で書いた re/im を別スレッドが phase 3 で読む。CUDA の __syncthreads() は
    // グローバルメモリも同期するが、Metal は mem_device を明示しないとデバイスメモリが見えない。
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // --- phase 3: 正規化「後」の値でハッシュ ------------------------------
    // 定数倍だけ違うブロックが同じキーになる必要があるので、必ず正規化後に取る。
    if (t == 0) { outCoef[0].zeroRe = 1u; outCoef[0].zeroIm = 1u; }   // 仮置き
    threadgroup_barrier(mem_flags::mem_device);
    ulong h = 0;
    for (uint i = t; i < total; i += tgsz) {
        if (re[i] != 0.0f) outCoef[0].zeroRe = 0u;   // 書く値は必ず 0 なので競合しても同じ
        if (im[i] != 0.0f) outCoef[0].zeroIm = 0u;
        ulong x = 1469598103934665603ul;
        x = fnv1a64_step(x, (ulong)i);
        x = fnv1a64_step(x, (ulong)as_type<uint>(re[i]));
        x = fnv1a64_step(x, (ulong)as_type<uint>(im[i]));
        h += x;   // 加算合成（XOR の打ち消しを避ける）
    }
    sh[t] = h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t == 0) {
        outCoef[0].re = nrm;
        outCoef[0].im = 0.0f;
        outId[0]      = sh[0];
    }
}

// ============================================================================
// バッチ（互いに独立な演算を 1 回の往復で処理する）
// Heian.cu の batch_*_kernel と同じ構成。threadgroup → ジョブ番号 を表（blockToOp）で引き、
// 全ジョブを 1 回の dispatch で回す。
// ジョブ記述子は入出力バッファを gpuAddress で直接持つ（Metal 3 が必要）。
// ★ Heian.mm の JobDescMetal / BatchOutHost と並び・型を一致させること
// ============================================================================
constant constexpr uint BATCH_TG = 256;

struct JobDesc {
    GPUInputHeader hdrA;           //   0
    GPUInputHeader hdrB;           //  28
    uint edgeOffA;                 //  56  エッジ連結バッファ内の先頭（QMDD のとき）
    uint edgeOffB;                 //  60
    uint dim;                      //  64  結果の一辺
    uint total;                    //  68  dim*dim
    uint firstBlock;               //  72  このジョブの先頭 threadgroup 番号
    uint partialOff;               //  76  部分和バッファ内のオフセット
    uint nBlocks;                  //  80
    uint _pad0;                    //  84  次のポインタを 8 バイト境界に揃える
    const device float* inReA;     //  88
    const device float* inImA;     //  96
    const device float* inReB;     // 104
    const device float* inImB;     // 112
    device float* outRe;           // 120
    device float* outIm;           // 128
};                                 // 136
static_assert(sizeof(JobDesc) == 136, "JobDesc の大きさが Heian.mm の JobDescMetal と合わない");

struct BatchOut {
    float coefRe;                  //  0
    float coefIm;                  //  4
    uint  zeroRe;                  //  8  実部が全要素ゼロなら 1
    uint  zeroIm;                  // 12
    ulong hash;                    // 16
};                                 // 24
static_assert(sizeof(BatchOut) == 24, "BatchOut の大きさが Heian.mm の BatchOutHost と合わない");

// 演算本体。opKind: 0=mul 1=add 2=kron（Heian.mm の BatchOpKind と同じ値）
kernel void batch_op(
    const device JobDesc* jobs      [[buffer(0)]],
    const device uint*    blockToOp [[buffer(1)]],
    const device GPUEdge* edges     [[buffer(2)]],
    constant uint&        opKind    [[buffer(3)]],
    uint bid                        [[threadgroup_position_in_grid]],
    uint lt                         [[thread_index_in_threadgroup]],
    uint tgsz                       [[threads_per_threadgroup]]
) {
    const uint op = blockToOp[bid];
    const device JobDesc& j = jobs[op];
    const uint tid = (bid - j.firstBlock) * tgsz + lt;
    if (tid >= j.total) return;
    const device GPUEdge* eA = edges + j.edgeOffA;
    const device GPUEdge* eB = edges + j.edgeOffB;

    float2 v = float2(0.0f, 0.0f);
    if (opKind == 0u) {                       // MUL
        const uint dim = j.dim;
        const uint row = tid / dim, col = tid - row * dim;
        float2 acc = float2(0.0f, 0.0f);
        for (uint k = 0; k < dim; ++k) {
            float2 a = evalInput(j.hdrA, eA, j.inReA, j.inImA, row, k, row * dim + k, false);
            float2 b = evalInput(j.hdrB, eB, j.inReB, j.inImB, k, col, k * dim + col, false);
            acc += cmul(a, b);
        }
        v = acc;
    } else if (opKind == 1u) {                // ADD
        const uint dim = j.dim;
        const uint row = tid / dim, col = tid - row * dim;
        float2 a = evalInput(j.hdrA, eA, j.inReA, j.inImA, row, col, tid, true);
        float2 b = evalInput(j.hdrB, eB, j.inReB, j.inImB, row, col, tid, true);
        v = a + b;
    } else {                                  // KRON
        // 実効次元（dim × pad）を使うこと。素の dim で割ると桁合わせが入ったオペランドで
        // 行・列の対応が崩れる（Heian.cu の batch_op_kernel と同じ注意）。
        const uint dimA = effDim(j.hdrA), dimB = effDim(j.hdrB);
        const uint dimOut = j.dim;
        const uint row = tid / dimOut, col = tid - row * dimOut;
        const uint rowA = row / dimB, colA = col / dimB;
        const uint rowB = row % dimB, colB = col % dimB;
        float2 a = evalInput(j.hdrA, eA, j.inReA, j.inImA, rowA, colA, rowA * dimA + colA, false);
        float2 b = evalInput(j.hdrB, eB, j.inReB, j.inImB, rowB, colB, rowB * dimB + colB, false);
        v = cmul(a, b);
    }
    j.outRe[tid] = v.x;
    j.outIm[tid] = v.y;
}

// threadgroup ごとの |v|^2 部分和
kernel void batch_sumsq(
    const device JobDesc* jobs      [[buffer(0)]],
    const device uint*    blockToOp [[buffer(1)]],
    device float*         partial   [[buffer(2)]],
    uint bid                        [[threadgroup_position_in_grid]],
    uint lt                         [[thread_index_in_threadgroup]],
    uint tgsz                       [[threads_per_threadgroup]]
) {
    const uint op = blockToOp[bid];
    const device JobDesc& j = jobs[op];
    const uint local = bid - j.firstBlock;
    const uint tid   = local * tgsz + lt;

    threadgroup float sd[BATCH_TG];
    float v = 0.0f;
    if (tid < j.total) v = j.outRe[tid] * j.outRe[tid] + j.outIm[tid] * j.outIm[tid];
    sd[lt] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz / 2; s > 0; s >>= 1) {
        if (lt < s) sd[lt] += sd[lt + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // local < nBlocks を確認してから書く（Heian.cu と同じ。詰め物の空ジョブ対策）
    if (lt == 0 && local < j.nBlocks) partial[j.partialOff + local] = sd[0];
}

// ジョブごとに部分和を集約して係数を決める（1 スレッド = 1 ジョブ）
kernel void batch_norm_finalize(
    const device JobDesc* jobs    [[buffer(0)]],
    const device float*   partial [[buffer(1)]],
    device BatchOut*      out     [[buffer(2)]],
    constant uint&        nJobs   [[buffer(3)]],
    uint op                       [[thread_position_in_grid]]
) {
    if (op >= nJobs) return;
    const device JobDesc& j = jobs[op];
    float s = 0.0f;
    for (uint i = 0; i < j.nBlocks; ++i) s += partial[j.partialOff + i];
    out[op].coefRe = (s > 0.0f) ? sqrt(s) : 0.0f;
    out[op].coefIm = 0.0f;
    out[op].zeroRe = 1u;      // 正規化後に非ゼロが見つかったら batch_hash_pass1 が下ろす
    out[op].zeroIm = 1u;
    out[op].hash   = 0ul;
}

kernel void batch_norm_apply(
    const device JobDesc*  jobs      [[buffer(0)]],
    const device uint*     blockToOp [[buffer(1)]],
    const device BatchOut* out       [[buffer(2)]],
    uint bid                         [[threadgroup_position_in_grid]],
    uint lt                          [[thread_index_in_threadgroup]],
    uint tgsz                        [[threads_per_threadgroup]]
) {
    const uint op = blockToOp[bid];
    const device JobDesc& j = jobs[op];
    const uint tid = (bid - j.firstBlock) * tgsz + lt;
    if (tid >= j.total) return;
    const float c = out[op].coefRe;
    if (c == 0.0f) return;
    float r = j.outRe[tid] / c;
    float m = j.outIm[tid] / c;
    if (fabs(r) < QMDD_GPU_ZERO_EPS) r = 0.0f;
    if (fabs(m) < QMDD_GPU_ZERO_EPS) m = 0.0f;
    j.outRe[tid] = r;
    j.outIm[tid] = m;
}

kernel void batch_hash_pass1(
    const device JobDesc* jobs      [[buffer(0)]],
    const device uint*    blockToOp [[buffer(1)]],
    device ulong*         partial   [[buffer(2)]],
    device BatchOut*      out       [[buffer(3)]],
    uint bid                        [[threadgroup_position_in_grid]],
    uint lt                         [[thread_index_in_threadgroup]],
    uint tgsz                       [[threads_per_threadgroup]]
) {
    const uint op = blockToOp[bid];
    const device JobDesc& j = jobs[op];
    const uint local = bid - j.firstBlock;
    const uint tid   = local * tgsz + lt;

    // 担当外のスレッドもバリアに到達させる（途中 return しない）
    threadgroup ulong sh[BATCH_TG];
    ulong h = 0;
    if (tid < j.total) {
        if (j.outRe[tid] != 0.0f) out[op].zeroRe = 0u;   // 書く値は必ず 0
        if (j.outIm[tid] != 0.0f) out[op].zeroIm = 0u;
        h = 1469598103934665603ul;
        h = fnv1a64_step(h, (ulong)tid);
        h = fnv1a64_step(h, (ulong)as_type<uint>(j.outRe[tid]));
        h = fnv1a64_step(h, (ulong)as_type<uint>(j.outIm[tid]));
    }
    sh[lt] = h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz / 2; s > 0; s >>= 1) {
        if (lt < s) sh[lt] += sh[lt + s];   // 加算合成: 順序非依存かつ打ち消しが起きにくい
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lt == 0 && local < j.nBlocks) partial[j.partialOff + local] = sh[0];
}

kernel void batch_hash_finalize(
    const device JobDesc* jobs    [[buffer(0)]],
    const device ulong*   partial [[buffer(1)]],
    device BatchOut*      out     [[buffer(2)]],
    constant uint&        nJobs   [[buffer(3)]],
    uint op                       [[thread_position_in_grid]]
) {
    if (op >= nJobs) return;
    const device JobDesc& j = jobs[op];
    ulong h = 0;
    for (uint i = 0; i < j.nBlocks; ++i) h += partial[j.partialOff + i];   // 部分和を加算合成
    out[op].hash = h;
}
