#include "backend.hpp"
#include "../common/parameter.hpp"

namespace {

static inline bool isZeroWeight(const complex<double>& w) {
    return w.real() == .0 && w.imag() == .0;
}

static inline void setRootWeight(GPUInput& out, const QMDDEdge& root) {
    out.root_re = static_cast<gpu_precision>(root.weight.real());
    out.root_im = static_cast<gpu_precision>(root.weight.imag());
}

// 子ノードを再帰的に詰めながら、各ノードの形（SHAPE_*）を求めて返す。
// level はこのノードの段（一辺 2^level）。終端の子は「残りの段の単位行列 × 重み」を表す。
static uint32_t fillNode(
    const QMDDNode* node,
    int32_t nodeIndex,
    int level,
    unordered_map<int64_t, int32_t>& keyToIndex,
    vector<GPUEdge>& outEdges,
    vector<uint32_t>& shapes
) {
    if (!node) return SHAPE_COL | SHAPE_DIAG;

    const size_t base = static_cast<size_t>(nodeIndex) * 4;
    if (outEdges.size() < base + 4) outEdges.resize(base + 4);

    uint32_t child[2][2];
    int k = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            const QMDDEdge& e = node->edges[i][j];

            GPUEdge ge;
            ge.re = static_cast<gpu_precision>(e.weight.real());
            ge.im = static_cast<gpu_precision>(e.weight.imag());

            if (isZeroWeight(e.weight)) {
                ge.childIndex = -1;
                child[i][j] = SHAPE_COL | SHAPE_DIAG | 4u;   // 4 = 零ブロックの印
            } else if (e.isTerminal || e.key_ == 0) {
                ge.childIndex = -1;
                // w·I（一辺 2^(level-1)）。1×1 ならスカラーなので列でもある
                child[i][j] = (level <= 1) ? (SHAPE_COL | SHAPE_DIAG) : SHAPE_DIAG;
            } else {
                const int64_t childKey = e.key_;

                auto it = keyToIndex.find(childKey);
                if (it == keyToIndex.end()) {

                    const int32_t newIndex = static_cast<int32_t>(keyToIndex.size());
                    keyToIndex.emplace(childKey, newIndex);

                    const size_t childBase = static_cast<size_t>(newIndex) * 4;
                    if (outEdges.size() < childBase + 4) outEdges.resize(childBase + 4);
                    if (shapes.size() <= static_cast<size_t>(newIndex)) shapes.resize(newIndex + 1, 0);

                    shapes[newIndex] = fillNode(e.node(), newIndex, level - 1, keyToIndex, outEdges, shapes);

                    ge.childIndex = newIndex;
                } else {
                    ge.childIndex = it->second;
                }
                child[i][j] = shapes[ge.childIndex];
            }

            outEdges[base + k] = ge;
            ++k;
        }
    }

    auto isZero = [](uint32_t c) { return (c & 4u) != 0; };
    uint32_t shape = 0;
    if (isZero(child[0][1]) && isZero(child[1][1]) && (child[0][0] & SHAPE_COL) && (child[1][0] & SHAPE_COL))
        shape |= SHAPE_COL;
    if (isZero(child[0][1]) && isZero(child[1][0]) && (child[0][0] & SHAPE_DIAG) && (child[1][1] & SHAPE_DIAG))
        shape |= SHAPE_DIAG;
    return shape;
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
    out.dim = 1u << root.level();

    unordered_map<int64_t, int32_t> keyToIndex;
    keyToIndex.reserve(1024);

    keyToIndex.emplace(root.key_, 0);

    out.qmdd.edges.resize(4);

    vector<uint32_t> shapes(1, 0);
    out.shape = fillNode(root.node(), 0, root.level(), keyToIndex, out.qmdd.edges, shapes);

    return out;
}

GPUInput wrapSV(const QMDDEdge& root) {

    GPUInput out;
    out.kind = SonKind::ArrayLeaf;
    setRootWeight(out, root);

    const ArrayLeaf* leaf = root.leaf();
    if (!leaf) {
        throw runtime_error("wrapSVEdge: edge does not reference ArrayLeaf.");
    }
    out.dim = static_cast<uint32_t>(leaf->dim());

    out.sv.reHandle = leaf->reBuf;
    out.sv.imHandle = leaf->imBuf;
    out.store = leaf->store;
    // 持ち方から分かる形だけを使う（GENERAL で持っている葉の中身までは調べない）
    out.shape = leaf->store == STORE_COLUMN   ? SHAPE_COL
              : leaf->store == STORE_DIAGONAL ? SHAPE_DIAG : 0u;

    return out;
}

GPUInput farewell(const QMDDEdge& root) {
    switch (root.sonKind_) {
        case SonKind::QMDDNode:
            return flattenQMDD(root);
        case SonKind::ArrayLeaf:
            return wrapSV(root);
        case SonKind::Terminal: {
            GPUInput out;
            out.kind = SonKind::Terminal;
            setRootWeight(out, root);
            out.shape = SHAPE_DIAG;   // w·I。一辺が 1 のときだけ列でもある（effShape で扱う）
            return out;
        }
        default:
            throw std::runtime_error("wrapAuto: unknown sonKind_");
    }
}

// pad（A ⊗ I_pad）を掛けたあとの形。対角は保たれ、列は一辺が 1 のときしか残らない。
static uint32_t effShape(const GPUInput& in) {
    uint32_t sh = in.shape;
    if (in.kind == SonKind::Terminal && effDim(in) <= 1) sh |= SHAPE_COL;
    if (in.pad > 1) sh &= SHAPE_DIAG;
    return sh;
}

uint32_t outputStore(GpuOp op, const GPUInput& A, const GPUInput& B) {
    const uint32_t a = effShape(A), b = effShape(B);
    uint32_t sh = 0;
    switch (op) {
        case GpuOp::Mul:  sh = (b & SHAPE_COL) | (a & b & SHAPE_DIAG); break;   // A·(列) は列
        case GpuOp::Add:  sh = a & b; break;
        case GpuOp::Kron: sh = a & b; break;
    }
    if (PARAMETER.leaf.column   && (sh & SHAPE_COL))  return STORE_COLUMN;
    if (PARAMETER.leaf.diagonal && (sh & SHAPE_DIAG)) return STORE_DIAGONAL;
    return STORE_GENERAL;
}
