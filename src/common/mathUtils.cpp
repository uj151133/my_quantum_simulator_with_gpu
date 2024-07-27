#include "mathUtils.hpp"

using namespace std;

QMDDEdge mathUtils::mul(const QMDDEdge& m, const QMDDEdge& v) {
    complex<double> weight = m.weight * v.weight;
    if (m.isTerminal && v.isTerminal) {
        return QMDDEdge(weight, nullptr);
    }

    QMDDNode* node = new QMDDNode(2);
    node->edges[0] = mathUtils::add(mathUtils::mul(m.node->edges[0], v.node->edges[0]),
                                    mathUtils::mul(m.node->edges[1], v.node->edges[1]));
    node->edges[1] = mathUtils::add(mathUtils::mul(m.node->edges[2], v.node->edges[0]),
                                    mathUtils::mul(m.node->edges[3], v.node->edges[1]));

    return QMDDEdge(weight, node);
}

QMDDEdge mathUtils::add(const QMDDEdge& edge1, const QMDDEdge& edge2) {
    // Aが終端ノードなら、Bのウェイトを加算して返す
    if (edge1.isTerminal) {
        QMDDEdge result = edge2;
        result.weight += edge1.weight;
        return result;
    }
    // Bが終端ノードなら、Aのウェイトを加算して返す
    if (edge2.isTerminal) {
        QMDDEdge result = edge1;
        result.weight += edge2.weight;
        return result;
    }

    // 子ノードのウェイトを親ノードのウェイトで掛け合わせる
    QMDDNode* newNode = new QMDDNode(4);
    for (int i = 0; i < 4; ++i) {
        // 再帰的に子ノードのエッジを加算
        QMDDEdge child1 = edge1.node->edges[i];
        QMDDEdge child2 = edge2.node->edges[i];
        
        // 子ノードのウェイトに親ノードのウェイトを掛け合わせる
        child1.weight *= edge1.weight;
        child2.weight *= edge2.weight;
        
        // 子ノードを加算
        newNode->edges[i] = mathUtils::add(child1, child2);
    }

    return QMDDEdge({1.0, 0.0}, newNode);
}
