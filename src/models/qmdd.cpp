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

CQMDDNode* convertToFFI(const shared_ptr<QMDDNode>& node) {
    if (!node || node->edges.empty()) {
        return nullptr;
    }
    
    uint32_t rows = node->edges.size();
    uint32_t cols = node->edges[0].size();
    uint32_t total_elements = rows * cols;
    
    CQMDDEdge* edges_array = new CQMDDEdge[total_elements];
    
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t col = 0; col < cols; col++) {
            uint32_t index = row * cols + col;
            const QMDDEdge& edge = node->edges[row][col];
            
            // QMDDEdge → CQMDDEdgeに変換
            edges_array[index] = {
                .weight = {
                    .real = edge.weight.real(),
                    .imag = edge.weight.imag()
                },
                .uniqueTableKey = edge.uniqueTableKey
            };
        }
    }
    
    CQMDDNode* cNode = new CQMDDNode{
        .edges_ptr = edges_array,
        .rows = rows,
        .cols = cols
    };
    
    return cNode;
}

shared_ptr<QMDDNode> convertFromFFI(const CQMDDNode& cNode) {
    if (cNode.edges_ptr == nullptr || cNode.rows == 0 || cNode.cols == 0) {
        return nullptr;
    }
    
    vector<vector<QMDDEdge>> edges(cNode.rows);
    
    for (uint32_t row = 0; row < cNode.rows; row++) {
        edges[row].reserve(cNode.cols);
        
        for (uint32_t col = 0; col < cNode.cols; col++) {
            uint32_t index = row * cNode.cols + col;
            const CQMDDEdge& cEdge = cNode.edges_ptr[index];
            
            QMDDEdge qEdge = QMDDEdge(
                complex<double>(cEdge.weight.real, cEdge.weight.imag),
                cEdge.uniqueTableKey
            );
            edges[row].push_back(qEdge);
        }
    }
    
    return make_shared<QMDDNode>(edges);
}

/////////////////////////////////////
//
//	QMDDEdge
//
/////////////////////////////////////

QMDDEdge::QMDDEdge(complex<double> w, shared_ptr<QMDDNode> n)
    : weight(w), uniqueTableKey((n && w != complex<double>(.0, .0)) ? calculation::generateUniqueTableKey(n) : 0), isTerminal(!n) {
    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif
    // if (this->uniqueTableKey) UniqueTable::getInstance().insert(this->uniqueTableKey, n);
    if (this->uniqueTableKey) insert(this->uniqueTableKey, convertToFFI(n));
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and uniqueTableKey: " << uniqueTableKey << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(double w, shared_ptr<QMDDNode> n)
    : weight(complex<double>(w, .0)), uniqueTableKey((n && w != .0) ? calculation::generateUniqueTableKey(n) : 0), isTerminal(!n) {
    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif
    // if (this->uniqueTableKey) UniqueTable::getInstance().insert(this->uniqueTableKey, n);
    if (this->uniqueTableKey) insert(this->uniqueTableKey, convertToFFI(n));
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and uniqueTableKey: " << uniqueTableKey << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(complex<double> w, int64_t key)
    : weight(w), uniqueTableKey(w != complex<double>(.0, .0) ? key : 0), isTerminal(key == 0) {
    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and uniqueTableKey: " << uniqueTableKey << " and isTerminal: " << isTerminal << endl;
}

QMDDEdge::QMDDEdge(double w, int64_t key)
    : weight(complex<double>(w, .0)), uniqueTableKey(w != .0 ? key : 0), isTerminal(key == 0) {
    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif
    this->calculateDepth();
    // cout << "Edge created with weight: " << weight << " and uniqueTableKey: " << uniqueTableKey << " and isTerminal: " << isTerminal << endl;
}

shared_ptr<QMDDNode> QMDDEdge::getStartNode() const {
    // return (uniqueTableKey == 0) ? nullptr : UniqueTable::getInstance().find(uniqueTableKey);
    return (this->uniqueTableKey == 0) ? nullptr : convertFromFFI(*find(this->uniqueTableKey));
}

vector<complex<double>> QMDDEdge::getAllElementsForKet() {
    vector<complex<double>> result;
    stack<pair<shared_ptr<QMDDNode>, size_t>> nodeStack;

    shared_ptr<QMDDNode> node = this->getStartNode();

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
    if (this->weight != other.weight) return false;
    if (this->isTerminal != other.isTerminal) return false;
    if (this->uniqueTableKey != other.uniqueTableKey) return false;
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

void QMDDEdge::calculateDepth() {
    if (this->isTerminal || this->uniqueTableKey == 0) {
        this->depth = 0;
    } else {
        shared_ptr<QMDDNode> startNode = this->getStartNode();
        
        vector<int> depths;
        for (const auto& edgeRow : startNode->edges) {
            for (const auto& edge : edgeRow) {
                depths.push_back(edge.depth);
            }
        }
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
    : initialEdge(std::move(edge)){}


shared_ptr<QMDDNode> QMDDGate::getStartNode() const {
    return this->initialEdge.getStartNode();
}

QMDDEdge QMDDGate::getInitialEdge() const {
    return this->initialEdge;
}

bool QMDDGate::operator==(const QMDDGate& other) const {
    return this->initialEdge == other.initialEdge;
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
    return this->initialEdge == other.initialEdge;
}

bool QMDDState::operator!=(const QMDDState& other) const {
    return !(*this == other);
}

ostream& operator<<(ostream& os, const QMDDState& state) {
    os << "QMDDState with initial edge:\n" << state.initialEdge;
    return os;
}

