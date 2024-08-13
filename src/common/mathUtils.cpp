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

    vector<QMDDEdge> newChildren(4);
    for (int i = 0; i < 4; ++i) {
        QMDDEdge child1 = (edge1.node && i < edge1.node->children.size()) ? edge1.node->children[i] : QMDDEdge(0.0, nullptr);
        QMDDEdge child2 = (edge2.node && i < edge2.node->children.size()) ? edge2.node->children[i] : QMDDEdge(0.0, nullptr);

        if (child1.node || child2.node) {
            child1.weight *= edge1.weight;
            child2.weight *= edge2.weight;
            newChildren[i] = multiplication(child1, child2);
        } else {
            newChildren[i] = QMDDEdge(0.0, nullptr);
        }
    }

    auto newNode = make_shared<QMDDNode>(newChildren);
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

    if (node1->children.size() != node2->children.size()) {
        throw std::runtime_error("Node edge sizes do not match.");
    }
    if (!node1 || !node2) {
        throw invalid_argument("Invalid node pointer in QMDDEdge.");
    }

    for (int i = 0; i < node1->children.size(); ++i) {
        node1->children[i].weight *= edge1.weight;
        node2->children[i].weight *= edge2.weight;
    }

    vector<QMDDEdge> newChildren(node1->children.size());
    for (size_t i = 0; i < newChildren.size(); ++i) {
        newChildren[i] = mathUtils::addition(node1->children[i], node2->children[i]);
    }

    auto newNode = make_shared<QMDDNode>(newChildren);
    return QMDDEdge(1.0, newNode);
}
