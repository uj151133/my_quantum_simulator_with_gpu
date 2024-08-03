#include "qmdd.hpp"
#include "uniqueTable.hpp"

using namespace std;

size_t QMDDNodeHashHelper::customHash(const std::complex<double>& c) const {
    size_t realHash = std::hash<double>()(c.real());
    size_t imagHash = std::hash<double>()(c.imag());
    std::cout << "customHash: real(" << c.real() << ") => " << realHash << ", imag(" << c.imag() << ") => " << imagHash << std::endl;
    return realHash ^ (imagHash << 1);
}

size_t QMDDNodeHashHelper::hashMatrixElement(const std::complex<double>& value, size_t row, size_t col) const {
    size_t valueHash = customHash(value);
    size_t hashValue = valueHash ^ (row + col + 0x9e3779b9 + (valueHash << 6) + (valueHash >> 2));
    std::cout << "hashMatrixElement: value(" << value << "), row(" << row << "), col(" << col << ") => " << hashValue << std::endl;
    return hashValue;
}

size_t QMDDNodeHashHelper::calculateMatrixHash(const QMDDNode& node, size_t row, size_t col, size_t rowStride, size_t colStride) const {
    size_t hashValue = 0;
    std::cout << "calculateMatrixHash: row(" << row << "), col(" << col << "), rowStride(" << rowStride << "), colStride(" << colStride << ")" << std::endl;
    for (size_t i = 0; i < node.edges.size(); ++i) {
        size_t newRow = row + (i / 2) * rowStride;
        size_t newCol = col + (i % 2) * colStride;

        if (!node.edges[i].node) {
            size_t elementHash = hashMatrixElement(node.edges[i].weight, newRow, newCol);
            std::cout << "Element hash at [" << newRow << "][" << newCol << "]: " << elementHash << std::endl;
            hashValue ^= elementHash;
        } else {
            size_t recursiveHash = calculateMatrixHash(*node.edges[i].node, newRow, newCol, rowStride * 2, colStride * 2);
            std::cout << "Recursive hash from node at [" << newRow << "][" << newCol << "]: " << recursiveHash << std::endl;
            hashValue ^= recursiveHash;
        }
    }
    std::cout << "Final hash value for node: " << hashValue << std::endl;
    return hashValue;
}

size_t QMDDNodeHashHelper::calculateMatrixHash(const QMDDNode& node) const {
    return calculateMatrixHash(node, 0, 0, 1, 1);
}


// QMDDEdgeのコンストラクタ
QMDDEdge::QMDDEdge(complex<double> w, shared_ptr<QMDDNode> n)
    : weight(w), node(n), isTerminal(!n) {
        cout << "Edge created with weight: " << weight << endl;
}

QMDDEdge::QMDDEdge(double w, shared_ptr<QMDDNode> n)
    : weight(complex<double>(w, 0.0)), node(n), isTerminal(!n) {
    cout << "Edge created with weight: " << weight << endl;
}
// QMDDEdgeの比較演算子
bool QMDDEdge::operator==(const QMDDEdge& other) const {
    return weight == other.weight && node == other.node;
}

ostream& operator<<(ostream& os, const QMDDEdge& edge) {
    os << "Weight: " << edge.weight;
    if (edge.node) {
        os << ", Node: " << *edge.node;
    } else {
        os << ", Node: Null";
    }
    return os;
}

// QMDDNodeのコンストラクタ
QMDDNode::QMDDNode(const vector<QMDDEdge>& edges) : edges(edges), uniqueTableKey(0) {
    QMDDNodeHashHelper hasher;
    uniqueTableKey = hasher.calculateMatrixHash(*this);
    
    std::cout << "Node created with " << edges.size() << " edges and uniqueTableKey: " << uniqueTableKey << std::endl;

    // UniqueTableの管理は必要に応じて実装してください
    // UniqueTable& table = UniqueTable::getInstance();
    // auto existingNode = table.findNode(uniqueTableKey, std::make_shared<QMDDNode>(*this));

    // if (existingNode == nullptr) {
    //     table.insertNode(uniqueTableKey, std::make_shared<QMDDNode>(*this));
    // } else {
    //     *this = *existingNode;
    // }
}



// ムーブ代入演算子: 適切な処理を確認
QMDDNode& QMDDNode::operator=(QMDDNode&& other) noexcept {
    if (this != &other) {
        edges = std::move(other.edges);
        // ムーブされた後のオブジェクトが安全に破棄されるようにする
        other.edges.clear();
    }
    return *this;
}

// QMDDNodeの比較演算子
bool QMDDNode::operator==(const QMDDNode& other) const {
    if (edges.size() != other.edges.size()) return false;
    for (size_t i = 0; i < edges.size(); ++i) {
        if (edges[i].weight != other.edges[i].weight) return false;
        if (edges[i].isTerminal != other.edges[i].isTerminal) return false;
        if (!edges[i].isTerminal && edges[i].node != other.edges[i].node) return false;
    }
    return true;
}


ostream& operator<<(ostream& os, const QMDDNode& node) {
    os << "QMDDNode with " << node.edges.size() << " edges and uniqueTableKey: " << node.uniqueTableKey << "\n";
    for (const auto& edge : node.edges) {
        os << "  " << edge << "\n";
    }
    return os;
}

// QMDDのコンストラクタ
QMDDGate::QMDDGate(QMDDEdge edge, size_t numEdges)
    : initialEdge(std::move(edge)) {
        if (initialEdge.node){
            initialEdge.node->edges.resize(numEdges);
        }
    }

// QMDDNodeの取得
QMDDNode* QMDDGate::getStartNode() const {
        return initialEdge.node.get();
    }

// QMDDEdgeの取得
QMDDEdge QMDDGate::getInitialEdge() const {
    return initialEdge;
}

ostream& operator<<(ostream& os, const QMDDGate& gate) {
    os << "QMDDGate with initial edge:\n" << gate.initialEdge;
    return os;
}

// QMDDStateのコンストラクタ
QMDDState::QMDDState(QMDDEdge edge, size_t numEdges)
    : initialEdge(std::move(edge)) {
    if (initialEdge.node) {
        initialEdge.node->edges.resize(numEdges);
    }
}


// QMDDStateのgetStartNodeメソッド
QMDDNode* QMDDState::getStartNode() const {
    return initialEdge.node.get();
}

// QMDDStateのgetInitialEdgeメソッド
QMDDEdge QMDDState::getInitialEdge() const {
    return initialEdge;
}

// QMDDStateのoperator+メソッド
QMDDState QMDDState::operator+(const QMDDState& other) {
    shared_ptr<QMDDNode> newNode = addNodes(this->getStartNode(), other.getStartNode());
    return QMDDState(QMDDEdge(this->initialEdge.weight + other.initialEdge.weight, newNode));
}

shared_ptr<QMDDNode> QMDDState::addNodes(QMDDNode* node1, QMDDNode* node2) {
        if (!node1) return shared_ptr<QMDDNode>(node2);
        if (!node2) return shared_ptr<QMDDNode>(node1);

        auto resultNode = make_shared<QMDDNode>(2);
        
        for (size_t i = 0; i < 2; ++i) {
            resultNode->edges[i].weight = node1->edges[i].weight + node2->edges[i].weight;
            resultNode->edges[i].node = addNodes(node1->edges[i].node.get(), node2->edges[i].node.get());
        }
        


        return resultNode;
    }

    ostream& operator<<(ostream& os, const QMDDState& state) {
    os << "QMDDState with initial edge:\n" << state.initialEdge;
    return os;
}

