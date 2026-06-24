#include "backend.hpp"
#include "../common/mathUtils.hpp"
#include <cstdio>
#include <cstdlib>

namespace {

static inline bool isZeroEdge(const QMDDEdge& edge) {
    return edge.magnitude == -numeric_limits<double>::infinity();
}

static inline void setRootWeight(GPUInput& out, const QMDDEdge& root) {
    const complex<double> w = mathUtils::toComplex(root.magnitude, root.angle);
    out.root_re = static_cast<gpu_precision>(w.real());
    out.root_im = static_cast<gpu_precision>(w.imag());
}

static void fillNode(
    const shared_ptr<QMDDNode>& node,
    int32_t nodeIndex,
    unordered_map<int64_t, int32_t>& keyToIndex,
    vector<GPUEdge>& outEdges
) {
    if (!node) return;

    const size_t base = static_cast<size_t>(nodeIndex) * 4;
    if (outEdges.size() < base + 4) outEdges.resize(base + 4);

    int k = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            const QMDDEdge& e = node->edges[i][j];

            GPUEdge ge;
            const complex<double> w = mathUtils::toComplex(e.magnitude, e.angle);
            ge.re = static_cast<gpu_precision>(w.real());
            ge.im = static_cast<gpu_precision>(w.imag());

            if (e.isTerminal || e.key_ == 0 || isZeroEdge(e)) {
                ge.childIndex = -1;
            } else {
                const int64_t childKey = e.key_;

                auto it = keyToIndex.find(childKey);
                if (it == keyToIndex.end()) {

                    const int32_t newIndex = static_cast<int32_t>(keyToIndex.size());
                    keyToIndex.emplace(childKey, newIndex);

                    const size_t childBase = static_cast<size_t>(newIndex) * 4;
                    if (outEdges.size() < childBase + 4) outEdges.resize(childBase + 4);

                    fillNode(get<shared_ptr<QMDDNode>>(e.getSon()), newIndex, keyToIndex, outEdges);

                    ge.childIndex = newIndex;
                } else {
                    ge.childIndex = it->second;
                }
            }

            outEdges[base + k] = ge;
            ++k;
        }
    }
}

}

GPUInput flattenQMDD(const QMDDEdge& root) {
    if (root.isTerminal || root.key_ == 0 || isZeroEdge(root)) {
        GPUInput out;
        out.kind = SonKind::Terminal;
        setRootWeight(out, root);
        return out;
    }

    GPUInput out;
    out.kind = SonKind::QMDDNode;
    setRootWeight(out, root);
    out.dim = 1u << root.depth;

    unordered_map<int64_t, int32_t> keyToIndex;
    keyToIndex.reserve(1024);

    keyToIndex.emplace(root.key_, 0);

    out.qmdd.edges.resize(4);

    fillNode(get<shared_ptr<QMDDNode>>(root.getSon()), 0, keyToIndex, out.qmdd.edges);

    return out;
}

// === DEBUG: SVLeaf の実部・虚部が両方とも全ゼロになっていないか検査する ============
// 1 にすると検査有効。GPU->host コピーが毎回入って非常に重い（マシンが落ちる）ので、
// 原因特定済みの今は 0（無効）。再調査したいときだけ 1 に戻すこと。
#ifndef SVLEAF_ZERO_CHECK
#define SVLEAF_ZERO_CHECK 0
#endif

#if SVLEAF_ZERO_CHECK
static bool bufferIsAllZero(void* handle, size_t total) {
    if (!handle || total == 0) return false;

    // まず先頭だけサンプル走査（普通のバッファはここで非ゼロが見つかり即終了）。
    const size_t sample = total < 4096 ? total : 4096;
    std::vector<float> host(sample, 0.0f);
    if (!copyGpuBufferToHostFloat(handle, host.data(), sample)) {
        // コピーできない場合はゼロ判定しない（誤検知防止）
        return false;
    }
    for (size_t i = 0; i < sample; ++i) {
        if (host[i] != 0.0f) return false;
    }
    if (sample == total) return true;

    // サンプルが全ゼロのときだけ全体を厳密確認。
    host.assign(total, 0.0f);
    if (!copyGpuBufferToHostFloat(handle, host.data(), total)) return false;
    for (size_t i = 0; i < total; ++i) {
        if (host[i] != 0.0f) return false;
    }
    return true;
}

static void warnIfZeroSVLeaf(const char* where, const QMDDEdge& root, const shared_ptr<SVLeaf>& leaf) {
    // 重みがゼロ（zero edge）なら中身が全ゼロでも正常（疎ブロック）なので無視。
    // 「重みは非ゼロなのに中身が全ゼロ」= 正規化的にあり得ない矛盾だけを拾う。
    if (isZeroEdge(root)) return;

    const size_t dim = leaf->dim;
    if (dim == 0) return;
    const size_t total = dim * dim;

    // まず実部だけ確認。実部に非ゼロがあれば「零行列」ではないので即終了（虚部コピー不要）。
    if (!bufferIsAllZero(leaf->reBuf, total)) return;
    // 実部が全ゼロのときだけ虚部を確認。
    if (!bufferIsAllZero(leaf->imBuf, total)) return;

    // ここに来たら「重み非ゼロなのに中身全ゼロ」= 正規化的にあり得ない矛盾。
    // 最初の1件で即停止する（フラッド＆過負荷でマシンが落ちるのを防ぐ）。
    // 直前の stdout の "Gate Idx: N" がそのまま犯人ゲート。
    fprintf(stderr,
        "\033[5;1;97;41m"
        "######## ZERO-BUFFER w/ NONZERO WEIGHT @%s "
        "mag=%g angle=%g leaf=%p re=%p im=%p dim=%zu use_count=%ld ######## (aborting)"
        "\033[0m\n",
        where,
        root.magnitude, root.angle,
        static_cast<void*>(leaf.get()),
        leaf->reBuf,
        leaf->imBuf,
        dim,
        leaf.use_count());
    fflush(stderr);
    std::abort();
}
#endif // SVLEAF_ZERO_CHECK

GPUInput wrapSV(const QMDDEdge& root) {

    GPUInput out;
    out.kind = SonKind::SVLeaf;
    setRootWeight(out, root);

    shared_ptr<SVLeaf> leaf = get<shared_ptr<SVLeaf>>(root.getSon());
    if (!leaf) {
        throw runtime_error("wrapSVEdge: edge does not reference SVLeaf.");
    }

#if SVLEAF_ZERO_CHECK
    warnIfZeroSVLeaf("wrapSV", root, leaf);
#endif

    out.dim = static_cast<uint32_t>(leaf->dim);

    out.sv.reHandle = leaf->reBuf;
    out.sv.imHandle = leaf->imBuf;

    return out;
}

GPUInput farewell(const QMDDEdge& root) {
    switch (root.sonKind_) {
        case SonKind::QMDDNode:
            return flattenQMDD(root);
        case SonKind::SVLeaf:
            return wrapSV(root);
        case SonKind::Terminal: {
            GPUInput out;
            out.kind = SonKind::Terminal;
            setRootWeight(out, root);
            return out;
        }
        default:
            throw std::runtime_error("wrapAuto: unknown sonKind_");
    }
}
