#include "backend.hpp"

#include <stdexcept>
#include <unordered_map>

using namespace std;

namespace {

static inline bool isZeroWeight(const complex<double>& w) {
    return w.real() == .0 && w.imag() == .0;
}

static inline void setRootWeight(GPUInput& out, const QMDDEdge& root) {
    out.root_re = static_cast<gpu_precision>(root.weight.real());
    out.root_im = static_cast<gpu_precision>(root.weight.imag());
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
            ge.re = static_cast<gpu_precision>(e.weight.real());
            ge.im = static_cast<gpu_precision>(e.weight.imag());

            if (e.isTerminal || e.key_ == 0 || isZeroWeight(e.weight)) {
                ge.childIndex = -1;
            } else {
                const int64_t childKey = e.key_;

                auto it = keyToIndex.find(childKey);
                if (it == keyToIndex.end()) {

                    const int32_t newIndex = static_cast<int32_t>(keyToIndex.size());
                    keyToIndex.emplace(childKey, newIndex);

                    const size_t childBase = static_cast<size_t>(newIndex) * 4;
                    if (outEdges.size() < childBase + 4) outEdges.resize(childBase + 4);

                    fillNode(e.getStartNode(), newIndex, keyToIndex, outEdges);

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
    if (root.isTerminal || root.key_ == 0 || isZeroWeight(root.weight)) {
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

    fillNode(root.getStartNode(), 0, keyToIndex, out.qmdd.edges);

    return out;
}

GPUInput wrapSV(const QMDDEdge& root) {

    GPUInput out;
    out.kind = SonKind::SVLeaf;
    setRootWeight(out, root);

    auto leaf = root.getStartLeaf();
    if (!leaf) {
        throw runtime_error("wrapSVEdge: edge does not reference SVLeaf.");
    }
    out.dim = static_cast<uint32_t>(leaf->dim);

    out.sv.reHandle = leaf->reBuf;
    out.sv.imHandle = leaf->imBuf;

    return out;
}

GPUInput farewell(const QMDDEdge& root) {
    if (root.isTerminal || root.key_ == 0 || isZeroWeight(root.weight)) {
        GPUInput out;
        out.kind = SonKind::Terminal;
        setRootWeight(out, root);
        return out;
    }

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