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

    // edge1.node と edge2.node の代わりに、uniqueTableKey からノードを取得
    std::shared_ptr<QMDDNode> node1 = table.find(edge1.uniqueTableKey);
    std::shared_ptr<QMDDNode> node2 = table.find(edge2.uniqueTableKey);

    if (edge1.isTerminal) {
        QMDDEdge result = edge2;
        result.weight += edge1.weight;
        cout << "Terminal edge1, returning result with weight: " << result.weight << endl;
        return result;
    }
    if (edge2.isTerminal) {
        QMDDEdge result = edge1;
        result.weight += edge2.weight;
        cout << "Terminal edge2, returning result with weight: " << result.weight << endl;
        return result;
    }

    if (!node1 || !node2) {
        cout << "Invalid node detected! node1: " << node1 << ", node2: " << node2 << endl;
        throw std::invalid_argument("Invalid node pointer in QMDDEdge.");
    }

    vector<QMDDEdge> newEdges(4);
    for (int i = 0; i < 4; ++i) {
        QMDDEdge child1 = (node1 && i < node1->edges.size()) ? node1->edges[i] : QMDDEdge(0.0, nullptr);
        QMDDEdge child2 = (node2 && i < node2->edges.size()) ? node2->edges[i] : QMDDEdge(0.0, nullptr);

        if (child1.node || child2.node) {
            newEdges[i] = QMDDEdge(0.0, nullptr);
        } else {
            child1.weight *= edge1.weight;
            child2.weight *= edge2.weight;
            newEdges[i] = mathUtils::addition(child1, child2);
        }
    }

    auto newNode = make_shared<QMDDNode>(newEdges);
    cout << "Created new node with uniqueTableKey: " << newNode->uniqueTableKey << endl;
    return QMDDEdge(1.0, newNode);
}
