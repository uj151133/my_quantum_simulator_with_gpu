#include "qmdd.hpp"
#include "uniqueTable.hpp"

using namespace std;

size_t QMDDNodeHashHelper::customHash(const std::complex<double>& c) const {
    size_t realHash = hash<double>()(c.real());
    size_t imagHash = hash<double>()(c.imag());
    // cout << "customHash: real(" << c.real() << ") => " << realHash << ", imag(" << c.imag() << ") => " << imagHash << endl;
    return realHash ^ (imagHash << 1);
}

size_t QMDDNodeHashHelper::calculateMatrixHash(const QMDDNode& node, size_t row, size_t col, size_t rowStride, size_t colStride, const complex<double>& parentWeight) const {
    size_t hashValue = 0;
    UniqueTable& table = UniqueTable::getInstance();

    for (size_t i = 0; i < node.edges.size(); ++i) {
        size_t newRow = row + (i / 2) * rowStride;
        size_t newCol = col + (i % 2) * colStride;

        complex<double> combinedWeight = parentWeight * node.edges[i].weight;

        size_t elementHash;
        if (node.edges[i].isTerminal || node.edges[i].uniqueTableKey == 0) {
            elementHash = hashMatrixElement(combinedWeight, newRow, newCol);
        } else {
            // find() の結果をデリファレンスして calculateMatrixHash に渡す
            shared_ptr<QMDDNode> foundNode = table.find(node.edges[i].uniqueTableKey);
            if (foundNode) {
                elementHash = calculateMatrixHash(*foundNode, newRow, newCol, rowStride * 2, colStride * 2, combinedWeight);
            } else {
                elementHash = 0;
            }
        }

        hashValue ^= (elementHash + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2));
    }

    return hashValue;
}


size_t QMDDNodeHashHelper::calculateMatrixHash(const QMDDNode& node) const {
    return calculateMatrixHash(node, 0, 0, 1, 1, complex<double>(1.0, 0.0));
}

size_t QMDDNodeHashHelper::hashMatrixElement(const complex<double>& value, size_t row, size_t col) const {
    size_t valueHash = customHash(value);
    size_t elementHash = valueHash ^ (row + col + 0x9e3779b9 + (valueHash << 6) + (valueHash >> 2));
    // cout << "hashMatrixElement: value(" << value << "), row(" << row << "), col(" << col << ") => " << elementHash << endl;
    return elementHash;
}

// QMDDEdgeのコンストラクタ
QMDDEdge::QMDDEdge(complex<double> w, shared_ptr<QMDDNode> n)
    : weight(w), uniqueTableKey(0), isTerminal(!n) {

    if (n) {
        uniqueTableKey = n->uniqueTableKey;
        cout << "Edge created with weight: " << weight << " and uniqueTableKey: " << uniqueTableKey << " and isTerminal: " << isTerminal<< endl;
        n.reset(); // ポインタを解放
    } else {
        cout << "Edge created with weight: " << weight << " (terminal node)" << endl;
    }
}


QMDDEdge::QMDDEdge(double w, shared_ptr<QMDDNode> n)
    : weight(complex<double>(w, 0.0)), uniqueTableKey(0), isTerminal(!n) {

    if (n) {
        uniqueTableKey = n->uniqueTableKey;
        // cout << "Edge created with weight: " << weight << " and uniqueTableKey: " << uniqueTableKey << endl;
        n.reset(); // ポインタを解放
    } else {
        // cout << "Edge created with weight: " << weight << " (terminal node)" << endl;
    }
}
// QMDDEdgeの比較演算子
bool QMDDEdge::operator==(const QMDDEdge& other) const {
    UniqueTable& table = UniqueTable::getInstance();
    return weight == other.weight && table.find(uniqueTableKey) == table.find(other.uniqueTableKey);
}

ostream& operator<<(ostream& os, const QMDDEdge& edge) {
    os << "Weight: " << edge.weight;
    if (edge.uniqueTableKey != 0) {
        os << ", Key: " << edge.uniqueTableKey;
    } else {
        os << ", Key: Null";
    }
    return os;
}

// QMDDNodeのコンストラクタ
QMDDNode::QMDDNode(const vector<QMDDEdge>& edges) : edges(edges), uniqueTableKey(0) {
    QMDDNodeHashHelper hasher;
    uniqueTableKey = hasher.calculateMatrixHash(*this);
    // cout << endl;
    // cout << "Node created with " << edges.size() << " edges and uniqueTableKey: " << uniqueTableKey << endl;

    UniqueTable& table = UniqueTable::getInstance();
    auto existingNode = table.check(uniqueTableKey, make_shared<QMDDNode>(*this));

    if (existingNode == nullptr) {
        table.insertNode(uniqueTableKey, make_shared<QMDDNode>(*this));
    } else {
        *this = *existingNode;
    }
    // cout << endl;
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
    UniqueTable& table = UniqueTable::getInstance();
    for (size_t i = 0; i < edges.size(); ++i) {
        if (edges[i].weight != other.edges[i].weight) return false;
        if (edges[i].isTerminal != other.edges[i].isTerminal) return false;
        if (!edges[i].isTerminal && edges[i].uniqueTableKey != other.edges[i].uniqueTableKey) return false;
        if (!edges[i].isTerminal && table.find(edges[i].uniqueTableKey) != table.find(other.edges[i].uniqueTableKey)) return false;
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
    : initialEdge(std::move(edge)), depth(0) {
    calculateDepth();
}

void QMDDGate::calculateDepth() {
    UniqueTable& table = UniqueTable::getInstance();
    auto currentNode = table.find(initialEdge.uniqueTableKey);
    size_t currentDepth = 0;

    while (currentNode && !currentNode->edges.empty()) {
        ++currentDepth;
        currentNode = table.find(currentNode->edges[0].uniqueTableKey); // 仮に最初のエッジをたどると仮定
    }

    cout << "Depth calculated: " << currentDepth << endl;
    depth = currentDepth;
}

// QMDDNodeの取得
QMDDNode* QMDDGate::getStartNode() const {
    UniqueTable& table = UniqueTable::getInstance();
    return table.find(initialEdge.uniqueTableKey).get();
}

// QMDDEdgeの取得
QMDDEdge QMDDGate::getInitialEdge() const {
    return initialEdge;
}

size_t QMDDGate::getDepth() const {
    return depth;
}

ostream& operator<<(ostream& os, const QMDDGate& gate) {
    os << "QMDDGate with initial edge:\n" << gate.initialEdge;
    return os;
}

// QMDDStateのコンストラクタ
QMDDState::QMDDState(QMDDEdge edge, size_t numEdges)
    : initialEdge(std::move(edge)) {
}

// QMDDStateのgetStartNodeメソッド
QMDDNode* QMDDState::getStartNode() const {
    UniqueTable& table = UniqueTable::getInstance();
    return table.find(initialEdge.uniqueTableKey).get();
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
    UniqueTable& table = UniqueTable::getInstance();
    if (!node1) return shared_ptr<QMDDNode>(node2);
    if (!node2) return shared_ptr<QMDDNode>(node1);

    vector<QMDDEdge> resultEdges = {
        QMDDEdge(node1->edges[0].weight + node2->edges[0].weight, addNodes(table.find(node1->edges[0].uniqueTableKey).get(), table.find(node2->edges[0].uniqueTableKey).get())),
        QMDDEdge(node1->edges[1].weight + node2->edges[1].weight, addNodes(table.find(node1->edges[1].uniqueTableKey).get(), table.find(node2->edges[1].uniqueTableKey).get()))
    };

    auto resultNode = make_shared<QMDDNode>(resultEdges);

    return resultNode;
}

    ostream& operator<<(ostream& os, const QMDDState& state) {
    os << "QMDDState with initial edge:\n" << state.initialEdge;
    return os;
}

