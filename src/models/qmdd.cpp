#include "qmdd.hpp"
#include "uniqueTable.hpp"
#include "../common/calculation.hpp"
#include "../common/mathUtils.hpp"

ostream& operator<<(ostream& os, const QMDDVariant& variant) {
    visit([&os](auto&& arg) {
        os << arg;
    }, variant);
    return os;
}

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

QMDDEdge::QMDDEdge(complex<double> w, shared_ptr<QMDDNode> n)
    : weight(w), key_((n && w != complex<double>(.0, .0)) ? calculation::generateUniqueTableKey(n) : 0), sonNode_(n), isTerminal(!n), sonKind_(SonKind::QMDDNode) {
    if (this->key_) UniqueTable::getInstance().insert(this->key_, this->sonNode_);
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(double w, shared_ptr<QMDDNode> n)
    : weight(complex<double>(w, .0)), key_((n && w != .0) ? calculation::generateUniqueTableKey(n) : 0), sonNode_(n), isTerminal(!n), sonKind_(SonKind::QMDDNode) {
    if (this->key_) UniqueTable::getInstance().insert(this->key_, this->sonNode_);
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(complex<double> w, int64_t key, shared_ptr<SVLeaf> l)
    : weight(w), key_(key), sonLeaf_(l), isTerminal(!l), sonKind_(SonKind::SVLeaf) {
    if (this->key_) Memo::getInstance().insert(this->key_, this->sonLeaf_);
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(double w, int64_t key, shared_ptr<SVLeaf> l)
    : weight(complex<double>(w, .0)), key_(key), sonLeaf_(l), isTerminal(!l), sonKind_(SonKind::SVLeaf) {
    // if (this->key_) Memo::getInstance().insert(this->key_, this->sonLeaf_);
    if (this->key_ != 0) {
        if (this->sonKind_ == SonKind::QMDDNode) {
            this->sonNode_ = UniqueTable::getInstance().find(this->key_);
            if (!this->sonNode_) {
                std::cerr << "UniqueTable miss: key=" << this->key_ << "\n";
            }
        } else {
            this->sonLeaf_ = Memo::getInstance().find(this->key_);
            if (!this->sonLeaf_) {
                std::cerr << "Memo miss: key=" << this->key_ << "\n";
            }
        }
    }
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(complex<double> w, int64_t key, SonKind kind)
    : weight(w), key_(w != complex<double>(.0, .0) ? key : 0), isTerminal(this->key_ == 0) {
    this->sonKind_ = kind;
    if (this->key_ != 0) {
        if (this->sonKind_ == SonKind::QMDDNode) {
            this->sonNode_ = UniqueTable::getInstance().find(this->key_);
        } else {
            this->sonLeaf_ = Memo::getInstance().find(this->key_);
        }
    }
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(double w, int64_t key, SonKind kind)
    : weight(complex<double>(w, .0)), key_(w != .0 ? key : 0), isTerminal(this->key_ == 0) {
    this->sonKind_ = kind;
    if (this->sonKind_ == SonKind::QMDDNode) {
        this->sonNode_ = UniqueTable::getInstance().find(this->key_);
    } else {
        this->sonLeaf_ = Memo::getInstance().find(this->key_);
    }
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and key_: " << key_ << " and isTerminal: " << isTerminal << endl;
}

shared_ptr<QMDDNode> QMDDEdge::getStartNode() const {
    return this->sonNode_;
}

shared_ptr<SVLeaf> QMDDEdge::getStartLeaf() const {
    return this->sonLeaf_;
}

vector<complex<double>> QMDDEdge::getAllElementsForKet() {
    vector<complex<double>> result;
    stack<pair<shared_ptr<QMDDNode>, size_t>> nodeStack;

    shared_ptr<QMDDNode> node = this->sonNode_;

    if (this->isTerminal) {
        result.push_back(weight);
    } else {
        nodeStack.push(make_pair(getStartNode(), 0));

        while (!nodeStack.empty()) {
            auto [node, edgeIndex] = nodeStack.top();
            nodeStack.pop();

            if (node->edges.size() == 1) {
                throw runtime_error("The start node has only one edge, which is not allowed.");
            }

            for (size_t i = edgeIndex; i < node->edges.size(); i++) {
                if (node->edges[i][0].isTerminal) {
                    result.push_back(node->edges[i][0].weight);
                } else {
                    nodeStack.push(make_pair(node, i + 1));
                    nodeStack.push(make_pair(node->edges[i][0].getStartNode(), 0));
                    break;
                }
            }
        }
    }
    return result;
}

bool QMDDEdge::operator==(const QMDDEdge& other) const {
    if (this->weight != other.weight) return false;
    if (this->isTerminal != other.isTerminal) return false;
    if (this->key_ != other.key_) return false;
    return true;
}

bool QMDDEdge::operator!=(const QMDDEdge& other) const {
    return !(*this == other);
}


ostream& operator<<(ostream& os, const QMDDEdge& edge) {
    os << "Weight = " << edge.weight;

    if (edge.key_ != 0) {
        os << ", Key = " << edge.key_ << ", isTerminal = " << edge.isTerminal;
    } else {
        os << ", Key = Null" << ", isTerminal = " << edge.isTerminal;
    }
    return os;
}

void QMDDEdge::calculateDepth() {
    if (this->isTerminal || this->key_ ==  0) {
        this->depth = 0;
    } else if (this->sonKind_ == SonKind::SVLeaf) {
        if (!this->sonLeaf_) {
            throw std::runtime_error("QMDDEdge::calculateDepth: sonLeaf_ is null");
        }
        this->depth = static_cast<int>(log2(this->sonLeaf_->dim));
    }
    else {
        vector<int> depths;
        for (const auto& edgeRow : this->getStartNode()->edges) {
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

