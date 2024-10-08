#include "gate.hpp"

static complex<double> i(0.0, 1.0);

QMDDGate gate::O() {
    return QMDDGate(QMDDEdge(.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    })));
}

QMDDGate gate::I() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    })));
}

QMDDGate gate::Ph(double delta) {
    return QMDDGate(QMDDEdge(exp(i * delta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    })));
}

QMDDGate gate::X() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    })));
}

// const QMDDGate gate::PLUS_X_GATE = [] {
//     complex<double> plusXWeight = 1 / sqrt(2.0);
//     auto plusXNode = make_shared<QMDDNode>(4);

//     plusXNode->children[0] = QMDDEdge(1, nullptr);
//     plusXNode->children[1] = QMDDEdge(i, nullptr);
//     plusXNode->children[2] = QMDDEdge(i, nullptr);
//     plusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusXEdge(plusXWeight, plusXNode);
//     return QMDDGate(plusXEdge);
// }();

// const QMDDGate gate::MINUS_X_GATE = [] {
//     complex<double> minusXWeight = 1 / sqrt(2.0);
//     auto minusXNode = make_shared<QMDDNode>(4);

//     minusXNode->children[0] = QMDDEdge(1, nullptr);
//     minusXNode->children[1] = QMDDEdge(-i, nullptr);
//     minusXNode->children[2] = QMDDEdge(-i, nullptr);
//     minusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusXEdge(minusXWeight, minusXNode);
//     return QMDDGate(minusXEdge);
// }();

QMDDGate gate::Y() {
    return QMDDGate(QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(-1.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    })));
}

// QMDDGate createPlusYGate() {
//     complex<double> plusYWeight = 1 / sqrt(2.0);
//     QMDDNode* plusYNode = new QMDDNode(4);

//     plusYNode->children[0] = QMDDEdge(1, nullptr);
//     plusYNode->children[1] = QMDDEdge(1, nullptr);
//     plusYNode->children[2] = QMDDEdge(-1, nullptr);
//     plusYNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusYEdge(plusYWeight, plusYNode);
//     return QMDDGate(plusYEdge);
// }

// QMDDGate createMinusYGate() {
//     complex<double> minusYWeight = 1 / sqrt(2.0);
//     QMDDNode* minusYNode = new QMDDNode(4);

//     minusYNode->children[0] = QMDDEdge(1, nullptr);
//     minusYNode->children[1] = QMDDEdge(-1, nullptr);
//     minusYNode->children[2] = QMDDEdge(1, nullptr);
//     minusYNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusYEdge(minusYWeight, minusYNode);
//     return QMDDGate(minusYEdge);
// }
QMDDGate gate::Z() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(-1.0, nullptr)}
    })));
}

QMDDGate gate::S() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(i, nullptr)}
    })));
}

QMDDGate gate::Sdagger() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(-i, nullptr)}
    })));
}

QMDDGate gate::V() {
    return QMDDGate(QMDDEdge(1.0 / 2.0 + i / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
        {QMDDEdge(i, nullptr), QMDDEdge(1.0, nullptr)}
    })));
}

QMDDGate gate::H() {
    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(-1.0, nullptr)}
    })));
}

QMDDGate gate::CX1() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))}
    })));
}

QMDDGate gate::CX2() {
    auto cx2Node1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto cx2Node2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, cx2Node1), QMDDEdge(1.0, cx2Node2)},
        {QMDDEdge(1.0, cx2Node2), QMDDEdge(1.0, cx2Node1)}
    })));
}

QMDDGate gate::varCX() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    })));
}

QMDDGate gate::CZ() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::Z().getStartNode()))}
    })));
}

QMDDGate gate::DCNOT() {
    auto dcnotNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto dcnotNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto dcnotNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    auto dcnotNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dcnotNode1), QMDDEdge(1.0, dcnotNode2)},
        {QMDDEdge(1.0, dcnotNode3), QMDDEdge(1.0, dcnotNode4)}
    })));
}

QMDDGate gate::SWAP(bool primitive) {
    if (primitive) {
        return QMDDGate(mathUtils::multiplication(mathUtils::multiplication(gate::CX1().getInitialEdge(), gate::CX2().getInitialEdge()), gate::CX1().getInitialEdge()));
    } else {
        auto swapNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        });

        auto swapNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        });

        auto swapNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        });

        auto swapNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
        });

        return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, swapNode1), QMDDEdge(1.0, swapNode2)},
            {QMDDEdge(1.0, swapNode3), QMDDEdge(1.0, swapNode4)}
        })));
    }
}

QMDDGate gate::iSWAP() {
    auto iswapNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto iswapNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto iswapNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto iswapNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, iswapNode1), QMDDEdge(i, iswapNode2)},
        {QMDDEdge(i, iswapNode3), QMDDEdge(1.0, iswapNode4)}
    })));
}

QMDDGate gate::P(double phi) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * phi), nullptr)}
    })));
}

QMDDGate gate::T() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * M_PI / 4.0), nullptr)}
    })));
}

QMDDGate gate::Tdagger() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(-i * M_PI / 4.0), nullptr)}
    })));
}

QMDDGate gate::CP(double phi) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::P(phi).getStartNode()))}
    })));
}

QMDDGate gate::CS() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::S().getStartNode()))}
    })));
}

QMDDGate gate::Rx(double theta) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(theta / 2.0), nullptr), QMDDEdge(-i * sin(theta / 2.0), nullptr)},
        {QMDDEdge(-i * sin(theta / 2.0), nullptr), QMDDEdge(cos(theta / 2.0), nullptr)}
    })));
}

QMDDGate gate::Ry(double theta) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(theta / 2.0), nullptr), QMDDEdge(-sin(theta / 2.0), nullptr)},
        {QMDDEdge(sin(theta / 2.0), nullptr), QMDDEdge(cos(theta / 2.0), nullptr)}
    })));
}

QMDDGate gate::Rz(double theta) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(exp(-i * theta / 2.0), nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * theta / 2.0), nullptr)}
    })));
}

QMDDGate gate::Rxx(double phi) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(-i * sin(phi / 2.0), make_shared<QMDDNode>(*gate::X().getStartNode()))},
        {QMDDEdge(-i * sin(phi / 2.0), make_shared<QMDDNode>(*gate::X().getStartNode())), QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(*gate::I().getStartNode()))}
    })));
}

QMDDGate gate::Ryy(double phi) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(-sin(phi / 2.0), make_shared<QMDDNode>(*gate::Y().getStartNode()))},
        {QMDDEdge(sin(phi / 2.0), make_shared<QMDDNode>(*gate::Y().getStartNode())), QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(*gate::I().getStartNode()))}
    })));
}

QMDDGate gate::Rzz(double phi) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::Rz(phi).getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(exp(i * phi / 2.0), make_shared<QMDDNode>(*gate::P(-phi).getStartNode()))}
    })));
}

QMDDGate gate::Rxy(double phi) {
    auto rxyNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(phi / 2.0), nullptr)}
    });

    auto rxyNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto rxyNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto rxyNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(phi / 2.0), nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, rxyNode1), QMDDEdge(-i * sin(phi / 2.0), rxyNode2)},
        {QMDDEdge(-i * sin(phi / 2.0), rxyNode3), QMDDEdge(1.0, rxyNode4)}
    })));
}

QMDDGate gate::SquareSWAP() {
    auto squareSWAPNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge((1.0 + i) / 2.0, nullptr)}
    });

    auto squareSWAPNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto squareSWAPNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto squareSWAPNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge((1.0 + i) / 2.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, squareSWAPNode1), QMDDEdge((1.0 - i) / 2.0, squareSWAPNode2)},
        {QMDDEdge((1.0 - i) / 2.0, squareSWAPNode3), QMDDEdge(1.0, squareSWAPNode4)}
    })));
}

QMDDGate gate::SquareiSWAP() {
    auto squareiSWAPNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0 / sqrt(2.0), nullptr)}
    });

    auto squareiSWAPNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto squareiSWAPNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto squareiSWAPNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0 / sqrt(2.0), nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, squareiSWAPNode1), QMDDEdge(i / sqrt(2.0), squareiSWAPNode2)},
        {QMDDEdge(i / sqrt(2.0), squareiSWAPNode3), QMDDEdge(1.0, squareiSWAPNode4)}
    })));
}

QMDDGate gate::SWAPalpha(double alpha) {
    auto SWAPalphaNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, nullptr)}
    });

    auto SWAPalphaNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto SWAPalphaNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto SWAPalphaNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, SWAPalphaNode1), QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode2)},
        {QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode3), QMDDEdge(1.0, SWAPalphaNode4)}
    })));
}

QMDDGate gate::FREDKIN() {
    auto fredkinNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    auto fredkinNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fredkinNode1), QMDDEdge(.0, fredkinNode2)},
        {QMDDEdge(.0, fredkinNode2), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::SWAP().getStartNode()))}
    })));
}

QMDDGate gate::U(double theta, double phi, double lambda) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(theta / 2.0), nullptr), QMDDEdge(-exp(i * lambda) * sin(theta / 2.0), nullptr)},
        {QMDDEdge(exp(i * phi) * sin(theta / 2.0), nullptr), QMDDEdge(exp(i * (lambda + phi)) * cos(theta / 2.0), nullptr)}
    })));
}

QMDDGate gate::BARENCO(double alpha, double phi, double theta) {
    auto barencoNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(exp(i * alpha) * cos(theta), nullptr), QMDDEdge(-i * exp(i * (alpha - phi)) * sin(theta), nullptr)},
        {QMDDEdge(-i * exp(i * (alpha + phi)) * sin(theta), nullptr), QMDDEdge(exp(i * alpha) * cos(theta), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, barencoNode1)}
    })));
}

QMDDGate gate::B() {
    auto bNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(M_PI / 8.0), nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(3.0 * M_PI / 8.0), nullptr)}
    });

    auto bNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(sin(M_PI / 8.0), nullptr)},
        {QMDDEdge(sin(3.0 * M_PI / 8.0), nullptr), QMDDEdge(.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, bNode1), QMDDEdge(i, bNode2)},
        {QMDDEdge(i * sin(M_PI / 8.0), make_shared<QMDDNode>(*gate::X().getStartNode())), QMDDEdge(cos(M_PI / 8.0), make_shared<QMDDNode>(*gate::I().getStartNode()))}
    })));
}

QMDDGate gate::CSX() {
    auto csxNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(exp(i * M_PI / 4.0), nullptr), QMDDEdge(exp(-i * M_PI / 4.0), nullptr)},
        {QMDDEdge(exp(-i * M_PI / 4.0), nullptr), QMDDEdge(exp(i * M_PI / 4.0), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, csxNode1)}
    })));
}

QMDDGate gate::N(double a, double b, double c) {
    auto nNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(exp(i * c) * cos(a - b), nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(-i * c) * cos(a + b), nullptr)}
    });

    auto nNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * c) * sin(a - b), nullptr)},
        {QMDDEdge(exp(-i * c) * sin(a + b), nullptr), QMDDEdge(.0, nullptr)}
    });

    auto nNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(-i * c) * sin(a + b), nullptr)},
        {QMDDEdge(exp(i * c) * sin(a - b), nullptr), QMDDEdge(.0, nullptr)}
    });

    auto nNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(exp(-i * c) * cos(a + b), nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * c) * cos(a - b), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nNode1), QMDDEdge(i, nNode2)},
        {QMDDEdge(i, nNode3), QMDDEdge(1.0, nNode4)}
    })));
}

QMDDGate gate::DB() {
    auto dbNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(3.0 * M_PI / 8.0), nullptr)}
    });

    auto dbNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto dbNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto dbNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(3.0 * M_PI / 8.0), nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dbNode1), QMDDEdge(-i * sin(3.0 * M_PI / 8.0), dbNode2)},
        {QMDDEdge(-i * sin(3.0 * M_PI / 8.0), dbNode3), QMDDEdge(1.0, dbNode4)}
    })));
}

QMDDGate gate::ECR() {
    auto ecrNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
        {QMDDEdge(i, nullptr), QMDDEdge(1.0, nullptr)}
    });

    auto ecrNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)},
        {QMDDEdge(-i, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, ecrNode1)},
        {QMDDEdge(1.0, ecrNode2), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    })));
}

QMDDGate gate::fSim(double theta, double phi) {
    auto fSimNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(theta), nullptr)}
    });

    auto fSimNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto fSimNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto fSimNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(theta), nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * phi), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fSimNode1), QMDDEdge(-i * sin(theta), fSimNode2)},
        {QMDDEdge(-i * sin(theta), fSimNode3), QMDDEdge(1.0, fSimNode4)}
    })));
}

QMDDGate gate::G(double theta) {
    auto gNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(theta), nullptr)}
    });

    auto gNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto gNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto gNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(theta), nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, gNode1), QMDDEdge(-sin(theta), gNode2)},
        {QMDDEdge(sin(theta), gNode3), QMDDEdge(1.0, gNode4)}
    })));
}

QMDDGate gate::M() {
    auto mNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto mNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(i, nullptr), QMDDEdge(1.0, nullptr)}
    });

    auto mNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)}
    });

    auto mNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(i, nullptr), QMDDEdge(-1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, mNode1), QMDDEdge(1.0, mNode2)},
        {QMDDEdge(1.0, mNode3), QMDDEdge(1.0, mNode4)}
    })));
}

QMDDGate gate::syc() {
    auto sycNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto sycNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto sycNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto sycNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, sycNode1), QMDDEdge(-i, sycNode2)},
        {QMDDEdge(-i, sycNode3), QMDDEdge(exp(-i * M_PI / 6.0), sycNode4)}
    })));
}

QMDDGate gate::CZS(double theta, double phi, double gamma) {
    auto czsNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(-exp(i * gamma) * std::pow(sin(theta / 2.0), 2) + std::pow(cos(theta / 2.0), 2), nullptr)}
    });

    auto czsNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto czsNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto czsNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(-exp(i * gamma) * std::pow(cos(theta / 2.0), 2) + std::pow(sin(theta / 2.0), 2), nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(-exp(i * gamma), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, czsNode1), QMDDEdge((1.0 + exp(i * gamma)) / 2.0 * exp(-i * phi) * sin(theta), czsNode2)},
        {QMDDEdge((1.0 + exp(i * gamma)) / 2.0 * exp(i * phi) * sin(theta), czsNode3), QMDDEdge(1.0, czsNode4)}
    })));
}

QMDDGate gate::D(double theta) {
    auto dNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    auto dNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    });

    auto dNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(i * cos(theta), nullptr), QMDDEdge(sin(theta), nullptr)},
        {QMDDEdge(sin(theta), nullptr), QMDDEdge(i * cos(theta), nullptr)}
    });

    auto dNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, dNode3)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dNode1), QMDDEdge(.0, dNode2)},
        {QMDDEdge(.0, dNode2), QMDDEdge(1.0, dNode4)}
    })));
}

QMDDGate gate::RCCX() {
    auto rccxNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    auto rccxNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    });

    auto rccxNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::Z().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, rccxNode1), QMDDEdge(.0, rccxNode2)},
        {QMDDEdge(.0, rccxNode2), QMDDEdge(1.0, rccxNode3)}
    })));
}

QMDDGate gate::PG() {
    auto pgNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    auto pgNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    });

    auto pgNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))},
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, pgNode1), QMDDEdge(.0, pgNode2)},
        {QMDDEdge(.0, pgNode2), QMDDEdge(1.0, pgNode3)}
    })));
}

QMDDGate gate::Toff() {
    auto toffNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    auto toffNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, toffNode1), QMDDEdge(.0, toffNode2)},
        {QMDDEdge(.0, toffNode2), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::CX1().getStartNode()))}
    })));
}

QMDDGate gate::fFredkin() {
    auto fFredkinNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    auto fFredkinNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    });

    auto fFredkinNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto fFredkinNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto fFredkinNode5 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    auto fFredkinNode6 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    auto fFredkinNode7 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fFredkinNode3), QMDDEdge(1.0, fFredkinNode4)},
        {QMDDEdge(1.0, fFredkinNode5), QMDDEdge(-1.0, fFredkinNode6)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fFredkinNode1), QMDDEdge(.0, fFredkinNode2)},
        {QMDDEdge(.0, fFredkinNode2), QMDDEdge(1.0, fFredkinNode7)}
    })));
}


// matrix Rotate(const ex &k){
//     return matrix{
//         {1, 0},
//         {0, exp((2 * Pi * I) / pow(2, k))}
//     };
// }

// matrix U1(const ex &lambda){
//     return matrix{
//         {1, 0},
//         {0, exp(I * lambda)}
//     };
// }

// matrix U2(const ex &phi, const ex &lambda){
//     return matrix{
//         {1, -exp(I * lambda)},
//         {exp(I * phi), exp(I * (lambda + phi))}
//     };
// }