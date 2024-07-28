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

    return QMDDEdge(weight, std::shared_ptr<QMDDNode>(node));
}

QMDDEdge mathUtils::add(const QMDDEdge& edge1, const QMDDEdge& edge2) {
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
            cout << "uuununu"<< endl;
            newNode->edges[i] = QMDDEdge(0.0, nullptr);
            
        } else {
            child1.weight *= edge1.weight;
            child2.weight *= edge2.weight;
            newNode->edges[i] = mathUtils::add(child1, child2);
        }
    }

    return QMDDEdge(1.0, newNode);
}
