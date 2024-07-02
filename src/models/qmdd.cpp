#include "qmdd.hpp"

using namespace std;

// QMDDEdgeのコンストラクタ
QMDDEdge::QMDDEdge(complex<double> w, QMDDNode* n)
    : weight(w), node(n), isTerminal(n == nullptr) {}


// QMDDEdgeの比較演算子
bool QMDDEdge::operator==(const QMDDEdge& other) const {
    return weight == other.weight && node == other.node;
}

// QMDDNodeのコンストラクタ
QMDDNode::QMDDNode(size_t numEdges){
    edges.resize(numEdges);
}

// QMDDNodeの比較演算子
bool QMDDNode::operator==(const QMDDNode& other) const {
    return edges == other.edges;
}

void QMDDNode::free() {
    for (auto& edge : edges) {
        if (edge.node != nullptr) {
            edge.node->free();
            delete edge.node;
            edge.node = nullptr;
        }
    }
}
// QMDDのコンストラクタ
QMDDGate::QMDDGate(QMDDEdge edge, size_t numEdges)
    : initialEdge(edge.weight, edge.node) {
        initialEdge.node->edges.resize(numEdges);
    }

// QMDDのデストラクタ
QMDDGate::~QMDDGate() {
    if (initialEdge.node != nullptr) {
            initialEdge.node->free();
            delete initialEdge.node;
            initialEdge.node = nullptr;
        }

}

// QMDDNodeの取得
QMDDNode* QMDDGate::getStartNode() const {
        return initialEdge.node;
    }

// QMDDEdgeの取得
QMDDEdge QMDDGate::getInitialEdge() const {
    return initialEdge;
}

// QMDDStateのコンストラクタ
QMDDState::QMDDState(QMDDEdge edge, size_t numEdges)
    : initialEdge(edge.weight, edge.node) {
    initialEdge.node->edges.resize(numEdges);
}

// QMDDStateのデストラクタ
QMDDState::~QMDDState() {
    if (initialEdge.node != nullptr) {
            initialEdge.node->free();
            delete initialEdge.node;
            initialEdge.node = nullptr;
        }
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


// mul関数
QMDDEdge mul(const QMDDEdge& m, const QMDDEdge& v) {
    complex<double> weight = m.weight * v.weight;
    if (m.isTerminal && v.isTerminal) {
        return QMDDEdge(weight, nullptr);
    }

    QMDDNode* node = new QMDDNode(2);
    node->edges[0] = add(mul(m.node->edges[0], v.node->edges[0]), mul(m.node->edges[1], v.node->edges[1]));
    node->edges[1] = add(mul(m.node->edges[2], v.node->edges[0]), mul(m.node->edges[3], v.node->edges[1]));

    return QMDDEdge(weight, node);
}


// add関数
QMDDEdge add(const QMDDEdge& e1, const QMDDEdge& e2) {
    if (!e1.node) return e2;
    if (!e2.node) return e1;

    complex<double> weight = e1.weight + e2.weight;
    QMDDNode* newNode = new QMDDNode(2);
    for (size_t i = 0; i < 2; ++i) {
        newNode->edges[i] = add(e1.node->edges[i], e2.node->edges[i]);
    }

    return QMDDEdge(weight, newNode);
}
