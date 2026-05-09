#include <metal_stdlib>
using namespace metal;

struct GPUEdge {
    float re;
    float im;
    int    childIndex; // -1 = terminal
};

struct GPUInputHeader {
    uint kind;     // 0=QMDD, 1=SV, 2=Terminal
    float root_re;
    float root_im;
    uint dim;      // sv.size（=行列の一辺）
};

struct NormVal {
    float re;
    float im;
};

inline bool isNonZero(float re, float im) {
    return (re != 0.0) || (im != 0.0);
}

kernel void norm_pass1(
    const device float* outRe      [[buffer(0)]],
    const device float* outIm      [[buffer(1)]],
    device uint* groupBestIdx       [[buffer(2)]],
    device NormVal* groupBestVal    [[buffer(3)]],
    constant uint& total                      [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]],
    uint tg_id                      [[threadgroup_position_in_grid]],
    uint tg_tid                     [[thread_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    uint idx = 0xFFFFFFFFu;
    NormVal v{0.0, 0.0};

    if (tid < total) {
        float re = outRe[tid];
        float im = outIm[tid];
        if (isNonZero(re, im)) {
            idx = tid;
            v = {re, im};
        }
    }

    threadgroup uint bestIdx[1024];
    threadgroup NormVal bestVal[1024];
    bestIdx[tg_tid] = idx;
    bestVal[tg_tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tg_tid < s) {
            uint idx2 = bestIdx[tg_tid + s];
            if (idx2 < bestIdx[tg_tid]) {
                bestIdx[tg_tid] = idx2;
                bestVal[tg_tid] = bestVal[tg_tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tg_tid == 0) {
        groupBestIdx[tg_id] = bestIdx[0];
        groupBestVal[tg_id] = bestVal[0];
    }
}

// pass2: 全groupの最小idxを選ぶ
kernel void norm_pass2(
    const device uint* groupBestIdx [[buffer(0)]],
    const device NormVal* groupBestVal [[buffer(1)]],
    device uint* outIndex           [[buffer(2)]],
    device NormVal* outCoef         [[buffer(3)]],
    constant uint& groupCount       [[buffer(4)]],
    uint tid                        [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint bestIdx = 0xFFFFFFFFu;
    NormVal bestVal{1.0, 0.0};

    for (uint i = 0; i < groupCount; ++i) {
        uint idx = groupBestIdx[i];
        if (idx < bestIdx) {
            bestIdx = idx;
            bestVal = groupBestVal[i];
        }
    }

    // 非ゼロが無かった場合
    if (bestIdx == 0xFFFFFFFFu) {
        bestIdx = 0xFFFFFFFFu;
        bestVal = {1.0, 0.0};
    }

    outIndex[0] = bestIdx;
    outCoef[0] = bestVal;
}

// pass3: 係数で割る（非ゼロが無かったら何もしない）
kernel void norm_apply(
    device float* outRe     [[buffer(0)]],
    device float* outIm     [[buffer(1)]],
    const device uint* idxBuf [[buffer(2)]],
    const device NormVal* coefBuf [[buffer(3)]],
    constant uint& total               [[buffer(4)]],
    uint tid                 [[thread_position_in_grid]]
) {
    if (tid >= total) return;
    if (idxBuf[0] == 0xFFFFFFFFu) return;

    float cr = coefBuf[0].re;
    float ci = coefBuf[0].im;

    // v / c = v * conj(c) / |c|^2
    float denom = cr*cr + ci*ci;
    float re = outRe[tid];
    float im = outIm[tid];

    float nr = (re*cr + im*ci) / denom;
    float ni = (im*cr - re*ci) / denom;

    outRe[tid] = nr;
    outIm[tid] = ni;
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
    uint tid                   [[thread_position_in_grid]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint tg_tid                [[thread_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    if (tid >= total) return;

    ulong h = 1469598103934665603ul;
    h = fnv1a64_step(h, (ulong)as_type<uint>(outRe[tid]));
    h = fnv1a64_step(h, (ulong)as_type<uint>(outIm[tid]));

    threadgroup ulong sh[1024];
    sh[tg_tid] = h;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tg_tid < s) {
            sh[tg_tid] ^= sh[tg_tid + s];
            sh[tg_tid] *= 1099511628211ul;
        }
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

    ulong h = 1469598103934665603ul;
    for (uint i = 0; i < partialCount; ++i) {
        h ^= partial[i];
        h *= 1099511628211ul;
    }
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

        if (e.childIndex == -1) break;
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
    if (hdr.kind == 1) { // SV
        float2 v = float2(inRe[tid], inIm[tid]);
        return cmul(float2(hdr.root_re, hdr.root_im), v);
    } else if (hdr.kind == 0) { // QMDD
        return evalDD(edges, float2(hdr.root_re, hdr.root_im), row, col, hdr.dim);
    } else { // Terminal
        return float2(0.0, 0.0);
    }
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
    uint dim = hdrA.dim;        // 前提：AとBで同じ
    uint total = dim * dim;
    if (tid >= total) return;

    uint row = tid / dim;
    uint col = tid - row * dim;

    float2 acc = float2(0.0, 0.0);

    for (uint k = 0; k < dim; ++k) {
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
    uint dim = hdrA.dim; // 前提：AとBで同じ
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
    uint dimA = hdrA.dim;
    uint dimB = hdrB.dim;
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

    float2 v = cmul(a, b);
    outRe[tid] = v.x;
    outIm[tid] = v.y;
}