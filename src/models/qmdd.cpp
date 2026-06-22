#include "qmdd.hpp"
#include "uniqueTable.hpp"
#include "../common/calculation.hpp"
#include "../common/mathUtils.hpp"

namespace {

double unity(double angle) {
    if (std::isnan(angle)) {
        return angle;
    }
    return std::remainder(angle, 2.0 * M_PI);
}

} // namespace

/////////////////////////////////////
//
//	QMDDEdge
//
/////////////////////////////////////
QMDDEdge::QMDDEdge() {}

QMDDEdge::QMDDEdge(const pair<double, double>& polar)
    : magnitude(polar.first), angle(unity(polar.second)) {}

QMDDEdge::QMDDEdge(const pair<double, double>& polar, SonVariant son)
    : magnitude(polar.first), angle(unity(polar.second)), son_(son) {
    this->Noah();
}

QMDDEdge::QMDDEdge(const pair<double, double>& polar, int64_t key, SonVariant son)
    : magnitude(polar.first), angle(unity(polar.second)), key_(key), son_(son) {
    this->Noah();
}

QMDDEdge::QMDDEdge(const pair<double, double>& polar, int64_t key, SonKind kind)
    : magnitude(polar.first), angle(unity(polar.second)), key_(polar.first != -std::numeric_limits<double>::infinity() ? key : 0), isTerminal(this->key_ == 0) {
    this->sonKind_ = kind;
    if (this->key_ != 0) {
        if (this->sonKind_ == SonKind::QMDDNode) {
            this->son_ = UniqueTable::getInstance().find(this->key_);
        } else {
            throw std::runtime_error("QMDDEdge(polar, key, SonKind::SVLeaf) is unsupported; pass shared_ptr<SVLeaf> instead");
        }
    }
    this->calculateDepth();
}

SonVariant QMDDEdge::getSon() const {
    return this->son_;
}

namespace {

complex<double> edgeCoeff(const QMDDEdge& edge) {
    if (edge.magnitude == -std::numeric_limits<double>::infinity()) {
        return {0.0, 0.0};
    }
    return polar(exp(edge.magnitude), edge.angle);
}

bool isZeroEdge(const QMDDEdge& edge) {
    return edge.magnitude == -std::numeric_limits<double>::infinity();
}

vector<complex<double>> readKetFromSVLeaf(const shared_ptr<SVLeaf>& leaf) {
    if (!leaf) throw runtime_error("readKetFromSVLeaf: leaf is null");
    const size_t dim = leaf->dim;
    if (dim == 0) return {};

    const size_t total = dim * dim;
    vector<float> re(total, 0.0f), im(total, 0.0f);

    if (!copyGpuBufferToHostFloat(leaf->reBuf, re.data(), total) ||
        !copyGpuBufferToHostFloat(leaf->imBuf, im.data(), total)) {
        throw runtime_error("readKetFromSVLeaf: failed to copy GPU buffer");
    }

    vector<complex<double>> ket(dim, {0.0, 0.0});
    for (size_t r = 0; r < dim; ++r) {
        const size_t idx = r * dim;
        ket[r] = {static_cast<double>(re[idx]), static_cast<double>(im[idx])};
    }
    return ket;
}

struct KetTask {
    QMDDEdge edge;
    complex<double> prefix;
    size_t base;
    size_t span;
};

} // namespace

vector<complex<double>> QMDDEdge::openKet() {
    if (this->isTerminal) {
        return {edgeCoeff(*this)};
    }

    if (this->depth < 0) {
        throw runtime_error("openKet: negative depth");
    }

    const size_t dim = static_cast<size_t>(1) << static_cast<size_t>(this->depth);
    vector<complex<double>> result(dim, {0.0, 0.0});

    stack<KetTask> st;
    st.push(KetTask{*this, {1.0, 0.0}, 0, dim});

    while (!st.empty()) {
        KetTask task = st.top();
        st.pop();

        const QMDDEdge& edge = task.edge;
        const complex<double> prefix = task.prefix * edgeCoeff(edge);

        if (edge.isTerminal) {
            result[task.base] += prefix;
            continue;
        }

        if (edge.sonKind_ == SonKind::SVLeaf) {
            const auto leaf = get<shared_ptr<SVLeaf>>(edge.getSon());
            const auto ket = readKetFromSVLeaf(leaf);

            if (ket.size() != task.span) {
                throw runtime_error("openKet: SVLeaf dim mismatch");
            }
            for (size_t i = 0; i < task.span; ++i) {
                result[task.base + i] += prefix * ket[i];
            }
            continue;
        }

        if (edge.sonKind_ != SonKind::QMDDNode) {
            throw runtime_error("openKet: unknown son kind");
        }

        const auto node = get<shared_ptr<QMDDNode>>(edge.getSon());
        if (!node) throw runtime_error("openKet: null QMDD node");

        if (node->edges.size() != 2 || node->edges[0].size() != 2 || node->edges[1].size() != 2) {
            throw runtime_error("openKet: expected 2x2 QMDD node");
        }

        if (!isZeroEdge(node->edges[0][1]) || !isZeroEdge(node->edges[1][1])) {
            throw runtime_error("openKet: non-zero second column (not ket form)");
        }

        if (task.span < 2 || (task.span & 1)) {
            throw runtime_error("openKet: invalid span");
        }

        const size_t half = task.span / 2;
        st.push(KetTask{node->edges[1][0], prefix, task.base + half, half});
        st.push(KetTask{node->edges[0][0], prefix, task.base, half});
    }

    return result;
}

bool QMDDEdge::operator==(const QMDDEdge& other) const {
    if (this->magnitude != other.magnitude) return false;
    if (this->angle != other.angle) return false;
    if (this->isTerminal != other.isTerminal) return false;
    if (this->key_ != other.key_) return false;
    if (this->sonKind_ != other.sonKind_) return false;
    return true;
}

bool QMDDEdge::operator!=(const QMDDEdge& other) const {
    return !(*this == other);
}


ostream& operator<<(ostream& os, const QMDDEdge& edge) {
    os << "Magnitude = " << edge.magnitude
        << ", Angle = " << edge.angle
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

QMDDSuite::QMDDSuite(const pair<double, double>& polar, SonVariant son)
    : initialEdge_(QMDDEdge(polar, son)) {}

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

