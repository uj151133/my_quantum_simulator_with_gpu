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
        throw std::invalid_argument("Invalid node pointer in QMDDEdge.");
    }

    auto newNode = make_shared<QMDDNode>(4);
    for (int i = 0; i < 4; ++i) {
        QMDDEdge child1 = (edge1.node && i < edge1.node->edges.size()) ? edge1.node->edges[i] : QMDDEdge(0.0, nullptr);
        QMDDEdge child2 = (edge2.node && i < edge2.node->edges.size()) ? edge2.node->edges[i] : QMDDEdge(0.0, nullptr);

        if (child1.node || child2.node) {
            child1.weight *= edge1.weight;
            child2.weight *= edge2.weight;
            newNode->edges[i] = multiplication(child1, child2);
        } else {
            newNode->edges[i] = QMDDEdge(0.0, nullptr);
        }
    }

    return QMDDEdge(1.0, newNode);
}

QMDDEdge mathUtils::addition(const QMDDEdge& edge1, const QMDDEdge& edge2) {
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

    if (!edge1.node || !edge2.node) {
        throw std::invalid_argument("Invalid node pointer in QMDDEdge.");
    }

    auto newNode = make_shared<QMDDNode>(4);
    for (int i = 0; i < 4; ++i) {
        QMDDEdge child1 = (edge1.node && i < edge1.node->edges.size()) ? edge1.node->edges[i] : QMDDEdge(0.0, nullptr);
        QMDDEdge child2 = (edge2.node && i < edge2.node->edges.size()) ? edge2.node->edges[i] : QMDDEdge(0.0, nullptr);

        if (child1.node || child2.node) {
            newNode->edges[i] = QMDDEdge(0.0, nullptr);
            
        } else {
            child1.weight *= edge1.weight;
            child2.weight *= edge2.weight;
            newNode->edges[i] = mathUtils::addition(child1, child2);
        }
    }

    return QMDDEdge(1.0, newNode);
}
