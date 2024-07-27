#include "qmdd.hpp"

using namespace std;

// QMDDEdgeのコンストラクタ
QMDDEdge::QMDDEdge(complex<double> w, QMDDNode* n)
    : weight(w), node(n), isTerminal(n == nullptr) {}

QMDDEdge::~QMDDEdge() {

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
QMDDNode::QMDDNode(size_t numEdges){
    edges.resize(numEdges);
}

QMDDNode::~QMDDNode() {
    for (auto& edge : edges) {
        delete edge.node;
        edge.node = nullptr;
    }
}
// ムーブコンストラクタ
QMDDNode::QMDDNode(QMDDNode&& other) noexcept
    : edges(move(other.edges)) {

    }

// ムーブ代入演算子
QMDDNode& QMDDNode::operator=(QMDDNode&& other) noexcept {
    if (this != &other) {
        for (auto& edge : edges) {
            delete edge.node;
        }
        edges = move(other.edges);
    }
    return *this;
}

// QMDDNodeの比較演算子
bool QMDDNode::operator==(const QMDDNode& other) const {
    return edges == other.edges;
}

ostream& operator<<(ostream& os, const QMDDNode& node) {
    os << "QMDDNode with " << node.edges.size() << " edges:\n";
    for (const auto& edge : node.edges) {
        os << "  " << edge << "\n";
    }
    return os;
}
// QMDDのコンストラクタ
QMDDGate::QMDDGate(QMDDEdge edge, size_t numEdges)
    : initialEdge(move(edge)) {
        if (initialEdge.node){
            initialEdge.node->edges.resize(numEdges);
        }
    }

// QMDDのデストラクタ
QMDDGate::~QMDDGate() {
    delete initialEdge.node;
    initialEdge.node = nullptr;
}

// QMDDNodeの取得
QMDDNode* QMDDGate::getStartNode() const {
        return initialEdge.node;
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
    : initialEdge(move(edge)) {
    if (initialEdge.node) {
        initialEdge.node->edges.resize(numEdges);
    }
}

// QMDDStateのデストラクタ
QMDDState::~QMDDState() {
    delete initialEdge.node;
    initialEdge.node = nullptr;
}

// QMDDStateのgetStartNodeメソッド
QMDDNode* QMDDState::getStartNode() const {
    return initialEdge.node;
}

// QMDDStateのgetInitialEdgeメソッド
QMDDEdge QMDDState::getInitialEdge() const {
    return initialEdge;
}

// QMDDStateのoperator+メソッド
QMDDState QMDDState::operator+(const QMDDState& other) {
    QMDDNode* newNode = addNodes(this->getStartNode(), other.getStartNode());
    return QMDDState(QMDDEdge(this->initialEdge.weight + other.initialEdge.weight, newNode));
}

QMDDNode* QMDDState::addNodes(QMDDNode* node1, QMDDNode* node2) {
        if (!node1) return node2;
        if (!node2) return node1;

        QMDDNode* resultNode = new QMDDNode(2);
        for (size_t i = 0; i < 2; ++i) {
            resultNode->edges[i].weight = node1->edges[i].weight + node2->edges[i].weight;
            resultNode->edges[i].node = addNodes(node1->edges[i].node, node2->edges[i].node);
        }

        return resultNode;
    }

    ostream& operator<<(ostream& os, const QMDDState& state) {
    os << "QMDDState with initial edge:\n" << state.initialEdge;
    return os;
}

