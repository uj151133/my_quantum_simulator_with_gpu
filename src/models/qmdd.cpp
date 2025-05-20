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

QMDDEdge::QMDDEdge(complex<double> w, shared_ptr<QMDDNode> n)
    : weight(w), uniqueTableKey(n ? calculation::generateUniqueTableKey(n) : 0), node(n), isTerminal(!n) {
    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif
    UniqueTable& table = UniqueTable::getInstance();
    if (n) {
        shared_ptr<QMDDNode> existingNode = table.find(uniqueTableKey);
        if (existingNode == nullptr) {
            table.insert(uniqueTableKey, n);
            this->node = n;
        } else {
            this->node = existingNode;
        }
    }
    this->depth = this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and uniqueTableKey: " << uniqueTableKey << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(double w, shared_ptr<QMDDNode> n)
    : weight(complex<double>(w, .0)), uniqueTableKey(n ? calculation::generateUniqueTableKey(n) : 0), node(n), isTerminal(!n) {
    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif
    UniqueTable& table = UniqueTable::getInstance();
    if (n) {
        auto existingNode = table.find(uniqueTableKey);
        if (existingNode == nullptr) {
            table.insert(uniqueTableKey, n);
            this->node = n;
        } else {
            this->node = existingNode;
        }
    }
    this->depth = this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and uniqueTableKey: " << uniqueTableKey << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(complex<double> w, long long key)
    : weight(w), uniqueTableKey(key), isTerminal(key == 0) {
    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif
    this->node = UniqueTable::getInstance().find(this->uniqueTableKey);
    this->depth = this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and uniqueTableKey: " << uniqueTableKey << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(double w, long long key)
    : weight(complex<double>(w, .0)), uniqueTableKey(key), isTerminal(key == 0) {
    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif
    this->node = UniqueTable::getInstance().find(this->uniqueTableKey);
    this->depth = this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and uniqueTableKey: " << uniqueTableKey << " and isTerminal: " << isTerminal << endl;
}

shared_ptr<QMDDNode> QMDDEdge::getStartNode() const {
    return this->node;
}

vector<complex<double>> QMDDEdge::getAllElementsForKet() {
    vector<complex<double>> result;
    stack<pair<shared_ptr<QMDDNode>, size_t>> nodeStack;

    if (isTerminal) {
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
    UniqueTable& table = UniqueTable::getInstance();
    if (weight != other.weight) return false;
    if (isTerminal != other.isTerminal) return false;
    if (!isTerminal && uniqueTableKey != other.uniqueTableKey) return false;
    if (!isTerminal && table.find(uniqueTableKey) != table.find(other.uniqueTableKey)) return false;
    return true;
}

bool QMDDEdge::operator!=(const QMDDEdge& other) const {
    return !(*this == other);
}


ostream& operator<<(ostream& os, const QMDDEdge& edge) {
    os << "Weight = " << edge.weight;

    if (edge.uniqueTableKey != 0) {
        os << ", Key = " << edge.uniqueTableKey << ", isTerminal = " << edge.isTerminal;
    } else {
        os << ", Key = Null" << ", isTerminal = " << edge.isTerminal;
    }
    return os;
}

int QMDDEdge::calculateDepth() const {
    if (this->isTerminal || this->uniqueTableKey ==  0 || this->node == nullptr) {
        return 0;
    } else {
        return 1 + this->node->edges[0][0].depth;
    }
}

/////////////////////////////////////
//
//	QMDDNode
//
/////////////////////////////////////

QMDDNode::QMDDNode(const vector<vector<QMDDEdge>>& edges) : edges(edges) {
}

QMDDNode& QMDDNode::operator=(QMDDNode&& other) noexcept {
    if (this != &other) {
        this->edges = std::move(other.edges);
        other.edges.clear();
    }
    return *this;
}

bool QMDDNode::operator==(const QMDDNode& other) const {
    if (edges.size() != other.edges.size()) return false;
    for (size_t i = 0; i < edges.size(); ++i) {
        if (edges[i].size() != other.edges[i].size()) return false;
        for (size_t j = 0; j < edges[i].size(); ++j) {
            if (edges[i][j] != other.edges[i][j]) return false;
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
    : initialEdge(std::move(edge)){}


shared_ptr<QMDDNode> QMDDGate::getStartNode() const {
    return this->initialEdge.getStartNode();
}

QMDDEdge QMDDGate::getInitialEdge() const {
    return this->initialEdge;
}

bool QMDDGate::operator==(const QMDDGate& other) const {
    return initialEdge == other.initialEdge;
}

bool QMDDGate::operator!=(const QMDDGate& other) const {
    return !(*this == other);
}

ostream& operator<<(ostream& os, const QMDDGate& gate) {
    os << "QMDDGate with initial edge:\n" << gate.initialEdge;
    return os;
}

/////////////////////////////////////
//
//	QMDDState
//
/////////////////////////////////////

QMDDState::QMDDState(QMDDEdge edge)
    : initialEdge(std::move(edge)) {}

shared_ptr<QMDDNode> QMDDState::getStartNode() const {
    return this->initialEdge.getStartNode();
}

QMDDEdge QMDDState::getInitialEdge() const {
    return this->initialEdge;
}

bool QMDDState::operator==(const QMDDState& other) const {
    return initialEdge == other.initialEdge;
}

bool QMDDState::operator!=(const QMDDState& other) const {
    return !(*this == other);
}

ostream& operator<<(ostream& os, const QMDDState& state) {
    os << "QMDDState with initial edge:\n" << state.initialEdge;
    return os;
}

