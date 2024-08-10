#include "mathUtils.hpp"

using namespace std;

QMDDEdge mathUtils::multiplication(const QMDDEdge& edge1, const QMDDEdge& edge2) {
    if (edge1.isTerminal) {
        QMDDEdge result = edge2;
        result.weight *= edge1.weight;
        return result;
    }
    if (edge2.isTerminal) {
        QMDDEdge result = edge1;
        result.weight *= edge2.weight;
        return result;
    }

    if (!edge1.node || !edge2.node) {
        throw invalid_argument("Invalid node pointer in QMDDEdge.");
    }

    vector<QMDDEdge> newEdges(4);
    for (int i = 0; i < 4; ++i) {
        QMDDEdge child1 = (edge1.node && i < edge1.node->edges.size()) ? edge1.node->edges[i] : QMDDEdge(0.0, nullptr);
        QMDDEdge child2 = (edge2.node && i < edge2.node->edges.size()) ? edge2.node->edges[i] : QMDDEdge(0.0, nullptr);

        if (child1.node || child2.node) {
            child1.weight *= edge1.weight;
            child2.weight *= edge2.weight;
            newEdges[i] = multiplication(child1, child2);
        } else {
            newEdges[i] = QMDDEdge(0.0, nullptr);
        }
    }

    auto newNode = make_shared<QMDDNode>(newEdges);
    return QMDDEdge(1.0, newNode);
}

QMDDEdge mathUtils::addition(const QMDDEdge& edge1, const QMDDEdge& edge2) {
    UniqueTable& table = UniqueTable::getInstance();

    shared_ptr<QMDDNode> node1 = table.find(edge1.uniqueTableKey);
    shared_ptr<QMDDNode> node2 = table.find(edge2.uniqueTableKey);

    if (edge1.isTerminal) {
        QMDDEdge result = edge2;
        result.weight += edge1.weight;
        return result;
    }
    if (edge2.isTerminal) {
        QMDDEdge result = edge1;
        result.weight += edge2.weight;
        return result;
    }

    if (!node1 || !node2) {
        throw invalid_argument("Invalid node pointer in QMDDEdge.");
    }

    // 子ノードの重みを掛け算する
    vector<QMDDEdge> newEdges(4);
    for (int i = 0; i < 4; ++i) {
        node1->edges[i].weight *= edge1.weight;
        node2->edges[i].weight *= edge2.weight;
    }

    // 子ノードの加算を再帰的に実行する
    newEdges[0] = mathUtils::addition(node1->edges[0], node2->edges[0]);
    newEdges[1] = mathUtils::addition(node1->edges[1], node2->edges[1]);
    newEdges[2] = mathUtils::addition(node1->edges[2], node2->edges[2]);
    newEdges[3] = mathUtils::addition(node1->edges[3], node2->edges[3]);

    auto newNode = make_shared<QMDDNode>(newEdges);
    return QMDDEdge(1.0, newNode);
}
