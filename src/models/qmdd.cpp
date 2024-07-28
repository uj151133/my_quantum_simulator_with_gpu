#include "qmdd.hpp"

using namespace std;

// QMDDEdgeのコンストラクタ
QMDDEdge::QMDDEdge(complex<double> w, shared_ptr<QMDDNode> n)
    : weight(w), node(n), isTerminal(!n) {}

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

// QMDDStateのデストラクタ
// QMDDState::~QMDDState() {
    // delete initialEdge.node.get();
    // initialEdge.node = nullptr;
// }

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

