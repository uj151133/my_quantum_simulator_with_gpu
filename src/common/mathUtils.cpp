#include "mathUtils.hpp"

using namespace std;

// QMDDEdge mathUtils::multiplication(const QMDDEdge& edge1, const QMDDEdge& edge2) {
//     if (edge1.isTerminal) {
//         QMDDEdge result = edge2;
//         result.weight *= edge1.weight;
//         return result;
//     }
//     if (edge2.isTerminal) {
//         QMDDEdge result = edge1;
//         result.weight *= edge2.weight;
//         return result;
//     }

//     if (!edge1.node || !edge2.node) {
//         throw invalid_argument("Invalid node pointer in QMDDEdge.");
//     }

//     vector<QMDDEdge> newEdge(4);
//     for (int i = 0; i < 4; ++i) {
//         QMDDEdge edge1 = (edge1.node && i < edge1.node->edges.size()) ? edge1.node->edges[i] : QMDDEdge(0.0, nullptr);
//         QMDDEdge edge2 = (edge2.node && i < edge2.node->edges.size()) ? edge2.node->edges[i] : QMDDEdge(0.0, nullptr);

//         if (edge1.node || edge2.node) {
//             edge1.weight *= edge1.weight;
//             edge2.weight *= edge2.weight;
//             newEdge[i] = multiplication(edge1, edge2);
//         } else {
//             newEdge[i] = QMDDEdge(0.0, nullptr);
//         }
//     }

//     auto newNode = make_shared<QMDDNode>(newEdge);
//     return QMDDEdge(1.0, newNode);
// }

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

    if (node1->edges.size() != node2->edges.size()) {
        throw std::runtime_error("Node edge sizes do not match.");
    }
    if (!node1 || !node2) {
        throw invalid_argument("Invalid node pointer in QMDDEdge.");
    }

    for (int i = 0; i < node1->edges.size(); ++i) {
        node1->edges[i].weight *= edge1.weight;
        node2->edges[i].weight *= edge2.weight;
    }

    vector<QMDDEdge> newEdge(node1->edges.size());
    for (size_t i = 0; i < newEdge.size(); ++i) {
        newEdge[i] = mathUtils::addition(node1->edges[i], node2->edges[i]);
    }

    auto newNode = make_shared<QMDDNode>(newEdge);
    return QMDDEdge(1.0, newNode);
}
