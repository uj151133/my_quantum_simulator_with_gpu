#include "qmdd.hpp"
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <atomic>
#include "uniqueTable.hpp"
#include "../common/calculation.hpp"
#include "../common/mathUtils.hpp"

/////////////////////////////////////
//
//	QMDDEdge
//
/////////////////////////////////////
QMDDEdge::QMDDEdge(){}

QMDDEdge::QMDDEdge(complex<double> w)
    :weight(w) {}

QMDDEdge::QMDDEdge(double w)
:weight(complex<double>(w, .0)) {}

static inline uintptr_t tagNode(QMDDNode* n) { return reinterpret_cast<uintptr_t>(n) | uintptr_t(1); }
static inline uintptr_t tagLeaf(ArrayLeaf* l)   { return reinterpret_cast<uintptr_t>(l) | uintptr_t(2); }

QMDDEdge::QMDDEdge(complex<double> w, QMDDNode* node)
    : weight(w), son_(node ? tagNode(node) : 0), sonKind_(node ? SonKind::QMDDNode : SonKind::Terminal), isTerminal(node == nullptr) {
    this->Noah();
}
QMDDEdge::QMDDEdge(double w, QMDDNode* node) : QMDDEdge(complex<double>(w, .0), node) {}

QMDDEdge::QMDDEdge(complex<double> w, ArrayLeaf* leaf)
    : weight(w), son_(leaf ? tagLeaf(leaf) : 0), sonKind_(leaf ? SonKind::ArrayLeaf : SonKind::Terminal), isTerminal(leaf == nullptr) {
    this->retain_();
    this->Noah();
}
QMDDEdge::QMDDEdge(double w, ArrayLeaf* leaf) : QMDDEdge(complex<double>(w, .0), leaf) {}

QMDDEdge::QMDDEdge(complex<double> w, int64_t key, QMDDNode* node)
    : weight(w), key_(key), son_(node ? tagNode(node) : 0), sonKind_(node ? SonKind::QMDDNode : SonKind::Terminal), isTerminal(node == nullptr) {
    this->Noah();
}
QMDDEdge::QMDDEdge(double w, int64_t key, QMDDNode* node) : QMDDEdge(complex<double>(w, .0), key, node) {}

QMDDEdge::QMDDEdge(complex<double> w, int64_t key, ArrayLeaf* leaf)
    : weight(w), key_(key), son_(leaf ? tagLeaf(leaf) : 0), sonKind_(leaf ? SonKind::ArrayLeaf : SonKind::Terminal), isTerminal(leaf == nullptr) {
    this->retain_();
    this->Noah();
}
QMDDEdge::QMDDEdge(double w, int64_t key, ArrayLeaf* leaf) : QMDDEdge(complex<double>(w, .0), key, leaf) {}

QMDDEdge::QMDDEdge(complex<double> w, int64_t key, SonKind kind)
    : weight(w), key_(w != complex<double>(.0, .0) ? key : 0), isTerminal(this->key_ == 0) {
    if (this->key_ == 0) { this->sonKind_ = SonKind::Terminal; return; }
    if (kind != SonKind::QMDDNode) {
        throw std::runtime_error("QMDDEdge(w, key, SonKind::ArrayLeaf) is unsupported; pass ArrayLeaf* instead");
    }
    QMDDNode* n = UniqueTable::getInstance().find(this->key_);   // 見つかれば参照 +1 済み
    if (!n) throw std::runtime_error("QMDDEdge(w, key, kind): node for key not found in unique table");
    this->sonKind_ = SonKind::QMDDNode;
    this->son_ = tagNode(n);
}
QMDDEdge::QMDDEdge(double w, int64_t key, SonKind kind) : QMDDEdge(complex<double>(w, .0), key, kind) {}


// vector<complex<double>> QMDDEdge::getAllElementsForKet() {
//     vector<complex<double>> result;
//     stack<pair<shared_ptr<QMDDNode>, size_t>> nodeStack;

//     if (this->isTerminal) {
//         result.push_back(weight);
//     } else {
//         nodeStack.push(make_pair(get<shared_ptr<QMDDNode>>(this->getSon()), 0));

//         while (!nodeStack.empty()) {
//             auto [node, edgeIndex] = nodeStack.top();
//             nodeStack.pop();

//             if (node->edges.size() == 1) {
//                 throw runtime_error("The start node has only one edge, which is not allowed.");
//             }

//             for (size_t i = edgeIndex; i < node->edges.size(); i++) {
//                 if (node->edges[i][0].isTerminal) {
//                     result.push_back(node->edges[i][0].weight);
//                 } else {
//                     nodeStack.push(make_pair(node, i + 1));
//                     nodeStack.push(make_pair(get<shared_ptr<QMDDNode>>(node->edges[i][0].getSon()), 0));
//                     break;
//                 }
//             }
//         }
//     }
//     return result;
// }

namespace {

    static std::vector<std::complex<double>> readKetFromArrayLeaf(const ArrayLeaf* leaf) {
        if (!leaf) throw std::runtime_error("readKetFromArrayLeaf: leaf is null");
        const size_t dim = leaf->dim();
        if (dim == 0) return {};

        const size_t total = dim * dim;
        std::vector<double> re(total, 0.0), im(total, 0.0);

        // 全要素ゼロの成分はバッファを持たない（容量削減）。その場合は 0 のまま。
        if (leaf->reBuf && !copyGpuBufferToHostDouble(leaf->reBuf, re.data(), total)) {
            throw std::runtime_error("readKetFromArrayLeaf: failed to copy GPU buffer");
        }
        if (leaf->imBuf && !copyGpuBufferToHostDouble(leaf->imBuf, im.data(), total)) {
            throw std::runtime_error("readKetFromArrayLeaf: failed to copy GPU buffer");
        }

        std::vector<std::complex<double>> ket(dim, {0.0, 0.0});
        for (size_t r = 0; r < dim; ++r) {
            const size_t idx = r * dim; // col=0
            ket[r] = {re[idx], im[idx]};
        }
        return ket;
    }

    struct KetTask {
        QMDDEdge edge;
        std::complex<double> prefix;
        size_t base;   // 出力先オフセット
        size_t span;   // このedgeが担当する要素数
    };
}

vector<std::complex<double>> QMDDEdge::openKet() {
    if (this->isTerminal) {
        return {this->weight};
    }

    const int lv = this->level();
    if (lv < 0) {
        throw std::runtime_error("getAllElementsForKet: negative level");
    }

    const size_t dim = static_cast<size_t>(1) << static_cast<size_t>(lv);
    std::vector<std::complex<double>> result(dim, {0.0, 0.0});

    std::stack<KetTask> st;
    st.push(KetTask{*this, {1.0, 0.0}, 0, dim});

    while (!st.empty()) {
        KetTask t = st.top();
        st.pop();

        const QMDDEdge& e = t.edge;
        const std::complex<double> nextPrefix = t.prefix * e.weight;

        if (e.isTerminal) {
            // identity 規約: カット終端 w は w·I_{span} を表す。
            // ket（列0）の読み出しでは I の列0 = e0 なので span 先頭の1要素だけ。
            result[t.base] += nextPrefix;
            continue;
        }

        if (e.sonKind_ == SonKind::ArrayLeaf) {
            auto ket = readKetFromArrayLeaf(e.leaf());

            if (ket.size() != t.span) {
                throw std::runtime_error("getAllElementsForKet: ArrayLeaf dim mismatch");
            }
            for (size_t i = 0; i < t.span; ++i) {
                result[t.base + i] += nextPrefix * ket[i];
            }
            continue;
        }

        if (e.sonKind_ != SonKind::QMDDNode) {
            throw std::runtime_error("getAllElementsForKet: unknown son kind");
        }

        const QMDDNode* node = e.node();
        if (!node) throw std::runtime_error("getAllElementsForKet: null QMDD node");

        if (node->edges.size() != 2 || node->edges[0].size() != 2 || node->edges[1].size() != 2) {
            throw std::runtime_error("getAllElementsForKet: expected 2x2 QMDD node");
        }

        // ket前提チェック（第2列はゼロ）
        if (!mathUtils::isZERO(node->edges[0][1].weight) || !mathUtils::isZERO(node->edges[1][1].weight)) {
            throw std::runtime_error("getAllElementsForKet: non-zero second column (not ket form)");
        }

        if (t.span < 2 || (t.span & 1)) {
            throw std::runtime_error("getAllElementsForKet: invalid span");
        }
        const size_t half = t.span / 2;

        // stackはLIFOなので、0側が先に処理されるよう 1->0 の順でpush
        st.push(KetTask{node->edges[1][0], nextPrefix, t.base + half, half});
        st.push(KetTask{node->edges[0][0], nextPrefix, t.base,        half});
    }

    return result;
}

bool QMDDEdge::operator==(const QMDDEdge& other) const {
    if (this->weight != other.weight) return false;
    if (this->isTerminal != other.isTerminal) return false;
    if (this->key_ != other.key_) return false;
    if (this->sonKind_ != other.sonKind_) return false;
    return true;
}

bool QMDDEdge::operator!=(const QMDDEdge& other) const {
    return !(*this == other);
}


ostream& operator<<(ostream& os, const QMDDEdge& edge) {
    os << "Weight = " << edge.weight
        << ", Key = " << (edge.key_ == 0 ? "Null" : std::to_string(edge.key_))
        << ", isTerminal = " << edge.isTerminal
        << ", SonKind = " << (edge.sonKind_ == SonKind::Terminal ? "Terminal" : (edge.sonKind_ == SonKind::QMDDNode ? "QMDDNode" : "ArrayLeaf"))
        << ", Level = " << edge.level()
        << "\n";

    // if (edge.sonKind_ == SonKind::QMDDNode) {
    //     os << "    SonNode: " << *edge.sonNode_ << "\n";
    // } else if (edge.sonKind_ == SonKind::ArrayLeaf) {
    //     os << "    SonLeaf: " << *edge.sonLeaf_  << "\n";
    // } else {
    //     os << "    NULL" << "\n";
    // }

    return os;
}

void QMDDEdge::Noah() {
    switch (this->sonKind_) {
        case SonKind::Terminal:
            this->isTerminal = true;
            this->key_ = 0;
            this->son_ = 0;
            return;
        case SonKind::QMDDNode: {
            QMDDNode* n = this->node();
            if (!n) throw std::runtime_error("QMDDNode child is null");
            this->isTerminal = false;
            if (this->key_ == 0) {
                this->key_ = (n->key != 0) ? n->key : calculation::generateUniqueTableKey(n);
            }
            // ユニークテーブルに登録。同内容が既にあればそれに差し替え、未登録の複製はここで捨てる。
            // insert は返すノードの参照カウントを +1 して返す（このエッジが所有する）。
            QMDDNode* canonical = UniqueTable::getInstance().insert(this->key_, n);
            if (canonical != n && n->key == 0) delete n;   // 未登録の複製はここで捨てる（子エッジの参照はデストラクタが返す）
            this->son_ = tagNode(canonical);
            return;
        }
        case SonKind::ArrayLeaf: {
            ArrayLeaf* l = this->leaf();
            if (!l) throw std::runtime_error("ArrayLeaf child is null");
            this->isTerminal = false;
            if (this->key_ == 0) {
                this->key_ = l->id != 0 ? l->id : static_cast<int64_t>(reinterpret_cast<uintptr_t>(l));
            }
            return;
        }
    }
}

// ---- 根の参照カウント ----
static std::atomic<bool> g_releaseDisabled{false};
void disableEdgeReleaseAtExit() { g_releaseDisabled.store(true, std::memory_order_relaxed); }

void QMDDEdge::retain_() const {
    const uintptr_t tag = this->son_ & uintptr_t(3);
    if (tag == uintptr_t(1)) {
        reinterpret_cast<QMDDNode*>(this->son_ & ~uintptr_t(3))->refCount.fetch_add(1, std::memory_order_relaxed);
    } else if (tag == uintptr_t(2)) {
        reinterpret_cast<ArrayLeaf*>(this->son_ & ~uintptr_t(3))->refCount.fetch_add(1, std::memory_order_relaxed);
    }
}
void QMDDEdge::release_() {
    if (g_releaseDisabled.load(std::memory_order_relaxed)) return;   // 終了処理中は解放しない（OS に任せる）
    const uintptr_t tag = this->son_ & uintptr_t(3);
    if (tag == uintptr_t(1)) {
        QMDDNode* n = reinterpret_cast<QMDDNode*>(this->son_ & ~uintptr_t(3));
        if (n->refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            UniqueTable::getInstance().release(n);   // テーブルから外して delete（他スレッドが拾い直していれば何もしない）
        }
    } else if (tag == uintptr_t(2)) {
        ArrayLeaf* l = reinterpret_cast<ArrayLeaf*>(this->son_ & ~uintptr_t(3));
        if (l->refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) delete l;   // GPU バッファはデストラクタが返す
    }
}

ArrayLeaf* pluck(size_t dim, void* reBuf, void* imBuf, int64_t id) {
    return new ArrayLeaf(dim, reBuf, imBuf, id);   // 所有は最初にこれを包む QMDDEdge へ
}

int QMDDEdge::level() const {
    switch (this->sonKind_) {
        case SonKind::Terminal: return 0;
        case SonKind::ArrayLeaf:   return this->leaf()->level;
        default:                return this->node()->level;
    }
}

/////////////////////////////////////
//
//	QMDDNode
//
/////////////////////////////////////

static int levelOf(const EdgeMatrix& e) {
    int m = 0;
    for (const auto& row : e) for (const auto& x : row) m = std::max(m, x.level());
    return 1 + m;
}

QMDDNode::QMDDNode(const EdgeMatrix& edges) : edges(edges), level(levelOf(this->edges)) {}
QMDDNode::QMDDNode(EdgeMatrix&& edges) noexcept : edges(std::move(edges)), level(levelOf(this->edges)) {}

QMDDNode::QMDDNode(const vector<vector<QMDDEdge>>& e) {
    if (e.size() != 2 || e[0].size() != 2 || e[1].size() != 2) {
        throw std::invalid_argument("QMDDNode: edges must be 2x2");
    }
    this->edges = {{ {e[0][0], e[0][1]}, {e[1][0], e[1][1]} }};
    this->level = levelOf(this->edges);
}

bool QMDDNode::operator==(const QMDDNode& other) const {
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (this->edges[i][j] != other.edges[i][j]) return false;
        }
    }
    return true;
}

bool QMDDNode::operator!=(const QMDDNode& other) const {
    return !(*this == other);
}

ostream& operator<<(ostream& os, const QMDDNode& node) {
    os << "Node with " << node.edges.size() << " rows of edges \n";
    for (size_t i = 0; i < node.edges.size(); i++) {
        for (size_t j = 0; j < node.edges[i].size(); j++) {
            const auto& edge = node.edges[i][j];
            os << "    Edge (" << i << ", " << j << "): " << edge << "\n";
        }
    }
    return os;
}

vector<complex<double>> QMDDNode::getWeights() const {
    vector<complex<double>> weights;
    for (const auto& edgeRow : this->edges) {
        for (const auto& edge : edgeRow) {
            weights.push_back(edge.weight);
        }
    }
    return weights;
}


/////////////////////////////////////
//
//	QMDDGate
//
/////////////////////////////////////

QMDDGate::QMDDGate(QMDDEdge edge)
    : initialEdge_(edge) {}

// shared_ptr<QMDDNode> QMDDGate::getStartNode() const {
//     return this->initialEdge_.getStartNode();
// }

QMDDEdge QMDDGate::getInitialEdge() const {
    return this->initialEdge_;
}

bool QMDDGate::operator==(const QMDDGate& other) const {
    return this->initialEdge_ == other.initialEdge_;
}

bool QMDDGate::operator!=(const QMDDGate& other) const {
    return !(*this == other);
}

ostream& operator<<(ostream& os, const QMDDGate& gate) {
    os << "QMDDGate with initial edge:\n" << gate.initialEdge_;
    return os;
}

/////////////////////////////////////
//
//	QMDDState
//
/////////////////////////////////////

QMDDState::QMDDState(QMDDEdge edge)
    : initialEdge_(edge) {}

// shared_ptr<QMDDNode> QMDDState::getStartNode() const {
//     return this->initialEdge_.getStartNode();
// }

QMDDEdge QMDDState::getInitialEdge() const {
    return this->initialEdge_;
}

bool QMDDState::operator==(const QMDDState& other) const {
    return this->initialEdge_ == other.initialEdge_;
}

bool QMDDState::operator!=(const QMDDState& other) const {
    return !(*this == other);
}

ostream& operator<<(ostream& os, const QMDDState& state) {
    os << "QMDDState with initial edge:\n" << state.initialEdge_;
    return os;
}


/////////////////////////////////////
//
//	QMDDSuite
//
/////////////////////////////////////

QMDDSuite::QMDDSuite(QMDDEdge edge)
    : initialEdge_(edge) {}

QMDDEdge QMDDSuite::getInitialEdge() const {
    return this->initialEdge_;
}

bool QMDDSuite::operator==(const QMDDSuite& other) const {
    return this->initialEdge_ == other.initialEdge_;
}

bool QMDDSuite::operator!=(const QMDDSuite& other) const {
    return !(*this == other);
}

ostream& operator<<(ostream& os, const QMDDSuite& suite) {
    os << "QMDDSuite with initial edge:\n" << suite.initialEdge_;
    return os;
}

