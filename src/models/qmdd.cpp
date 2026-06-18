#include "qmdd.hpp"
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

QMDDEdge::QMDDEdge(complex<double> w, SonVariant son)
    : weight(w), son_(son) {
    this->Noah();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(double w, SonVariant son)
    : weight(complex<double>(w, .0)), son_(son){
    this->Noah();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(complex<double> w, int64_t key, SonVariant son)
    : weight(w), key_(key), son_(son) {
    this->Noah();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(double w, int64_t key, SonVariant son)
    : weight(complex<double>(w, .0)), key_(key), son_(son) {
    this->Noah();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(complex<double> w, int64_t key, SonKind kind)
    : weight(w), key_(w != complex<double>(.0, .0) ? key : 0), isTerminal(this->key_ == 0) {
    this->sonKind_ = kind;
    if (this->key_ != 0) {
        if (this->sonKind_ == SonKind::QMDDNode) {
            this->son_ = UniqueTable::getInstance().find(this->key_);
        } else {
            this->son_ = Memo::getInstance().find(this->key_);
        }
    }
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(double w, int64_t key, SonKind kind)
    : weight(complex<double>(w, .0)), key_(w != .0 ? key : 0), isTerminal(this->key_ == 0) {
    if (this->key_ != 0) this->sonKind_ = kind;
    if (this->sonKind_ == SonKind::QMDDNode) {
        this->son_ = UniqueTable::getInstance().find(this->key_);
    } else {
        this->son_ = Memo::getInstance().find(this->key_);
    }
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

// shared_ptr<QMDDNode> QMDDEdge::getStartNode() const {
//     return this->sonNode_;
// }

// shared_ptr<SVLeaf> QMDDEdge::getStartLeaf() const {
//     return this->sonLeaf_;
// }

SonVariant QMDDEdge::getSon() const {
    return this->son_;
}

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

    static std::vector<std::complex<double>> readKetFromSVLeaf(const std::shared_ptr<SVLeaf>& leaf) {
        if (!leaf) throw std::runtime_error("readKetFromSVLeaf: leaf is null");
        const size_t dim = leaf->dim;
        if (dim == 0) return {};

        const size_t total = dim * dim;
        std::vector<float> re(total, 0.0f), im(total, 0.0f);

        if (!copyGpuBufferToHostFloat(leaf->reBuf, re.data(), total) ||
            !copyGpuBufferToHostFloat(leaf->imBuf, im.data(), total)) {
            throw std::runtime_error("readKetFromSVLeaf: failed to copy GPU buffer");
        }

        std::vector<std::complex<double>> ket(dim, {0.0, 0.0});
        for (size_t r = 0; r < dim; ++r) {
            const size_t idx = r * dim; // col=0
            ket[r] = {static_cast<double>(re[idx]), static_cast<double>(im[idx])};
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

    if (this->depth < 0) {
        throw std::runtime_error("getAllElementsForKet: negative depth");
    }

    const size_t dim = static_cast<size_t>(1) << static_cast<size_t>(this->depth);
    std::vector<std::complex<double>> result(dim, {0.0, 0.0});

    std::stack<KetTask> st;
    st.push(KetTask{*this, {1.0, 0.0}, 0, dim});

    while (!st.empty()) {
        KetTask t = st.top();
        st.pop();

        const QMDDEdge& e = t.edge;
        const std::complex<double> nextPrefix = t.prefix * e.weight;
        // std::cerr
        //     << "[openKet] pop"
        //     << " span=" << t.span
        //     << " base=" << t.base
        //     << " depth=" << e.depth
        //     << " isTerminal=" << e.isTerminal
        //     << " sonKind=" << static_cast<int>(e.sonKind_)
        //     << " key=" << e.key_
        //     << " weight=(" << e.weight.real() << "," << e.weight.imag() << ")\n";

        if (e.isTerminal) {
            // identity 規約: カット終端 w は w·I_{span} を表す。
            // ket（列0）の読み出しでは I の列0 = e0 なので span 先頭の1要素だけ。
            result[t.base] += nextPrefix;
            continue;
        }

        if (e.sonKind_ == SonKind::SVLeaf) {
            auto leaf = get<shared_ptr<SVLeaf>>(e.getSon());
            auto ket = readKetFromSVLeaf(leaf);

            if (ket.size() != t.span) {
                throw std::runtime_error("getAllElementsForKet: SVLeaf dim mismatch");
            }
            for (size_t i = 0; i < t.span; ++i) {
                result[t.base + i] += nextPrefix * ket[i];
            }
            continue;
        }

        if (e.sonKind_ != SonKind::QMDDNode) {
            throw std::runtime_error("getAllElementsForKet: unknown son kind");
        }

        auto node = std::get<std::shared_ptr<QMDDNode>>(e.getSon());
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
        // std::cerr
        //     << "[openKet] split"
        //     << " parent_span=" << t.span
        //     << " half=" << half
        //     << " left(depth=" << node->edges[0][0].depth << ", key=" << node->edges[0][0].key_ << ")"
        //     << " right(depth=" << node->edges[1][0].depth << ", key=" << node->edges[1][0].key_ << ")\n";

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
        << ", SonKind = " << (edge.sonKind_ == SonKind::Terminal ? "Terminal" : (edge.sonKind_ == SonKind::QMDDNode ? "QMDDNode" : "SVLeaf"))
        << ", Depth = " << edge.depth
        << "\n";

    // if (edge.sonKind_ == SonKind::QMDDNode) {
    //     os << "    SonNode: " << *edge.sonNode_ << "\n";
    // } else if (edge.sonKind_ == SonKind::SVLeaf) {
    //     os << "    SonLeaf: " << *edge.sonLeaf_  << "\n";
    // } else {
    //     os << "    NULL" << "\n";
    // }

    return os;
}

void QMDDEdge::Noah() {
    visit([&](auto const& child) {
        using T = decay_t<decltype(child)>;
        if constexpr (std::is_same_v<T, monostate>) {
            this->sonKind_ = SonKind::Terminal;
            this->isTerminal = true;
            this->key_ = 0;
        } else if constexpr (std::is_same_v<T, shared_ptr<QMDDNode>>) {
            if (!child) throw std::runtime_error("QMDDNode child is null");
            this->sonKind_ = SonKind::QMDDNode;
            this->isTerminal = false;
            if (this->key_ == 0) {
                this->key_ = calculation::generateUniqueTableKey(child);
            }
            UniqueTable::getInstance().insert(this->key_, child);
            this->calculateDepth();
        } else if constexpr (std::is_same_v<T, shared_ptr<SVLeaf>>) {
            if (!child) throw std::runtime_error("SVLeaf child is null");
            this->sonKind_ = SonKind::SVLeaf;
            this->isTerminal = false;
            Memo::getInstance().insert(this->key_, child);
            this->calculateDepth();
        }
    }, this->son_);
}

void QMDDEdge::calculateDepth() {
    if (this->sonKind_ == SonKind::Terminal) {
        this->depth = 0;
    } else if (this->sonKind_ == SonKind::SVLeaf) {
        if (holds_alternative<monostate>(this->son_)) {
            throw std::runtime_error("QMDDEdge::calculateDepth: sonLeaf_ is null");
        }
        this->depth = static_cast<int>(log2(get<shared_ptr<SVLeaf>>(this->son_)->dim));
    }
    else {
        vector<int> depths;
        for (const auto& edgeRow : get<shared_ptr<QMDDNode>>(this->son_)->edges) {
            for (const auto& edge : edgeRow) {
                depths.push_back(edge.depth);
            }
        }
        // return 1 + this->getStartNode()->edges[0][0].depth;
        this->depth = 1 + *max_element(depths.begin(), depths.end());
    }
    return;
}

/////////////////////////////////////
//
//	QMDDNode
//
/////////////////////////////////////

QMDDNode::QMDDNode(const vector<vector<QMDDEdge>>& edges) : edges(edges) {}

QMDDNode& QMDDNode::operator=(QMDDNode&& other) noexcept {
    if (this != &other) {
        this->edges = std::move(other.edges);
        other.edges.clear();
    }
    return *this;
}

bool QMDDNode::operator==(const QMDDNode& other) const {
    if (this->edges.size() != other.edges.size()) return false;
    for (size_t i = 0; i < edges.size(); ++i) {
        if (this->edges[i].size() != other.edges[i].size()) return false;
        for (size_t j = 0; j < edges[i].size(); ++j) {
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
    for (int i = 0; i < node.edges.size(); i++) {
        for (int j = 0; j < node.edges[i].size(); j++) {
            auto edge = node.edges[i][j];
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
    : initialEdge_(edge){}

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

