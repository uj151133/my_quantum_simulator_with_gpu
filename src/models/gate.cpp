#include "gate.hpp"

static complex<double> i(0.0, 1.0);

QMDDGate gate::O() {
    // return QMDDGate(QMDDEdge(.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
    //     {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
    //     {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    // })));
    return QMDDGate(QMDDEdge(.0, nullptr));
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
//     shared_ptr<QMDDNode> plusXNode = make_shared<QMDDNode>(4);

//     plusXNode->children[0] = QMDDEdge(1, nullptr);
//     plusXNode->children[1] = QMDDEdge(i, nullptr);
//     plusXNode->children[2] = QMDDEdge(i, nullptr);
//     plusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusXEdge(plusXWeight, plusXNode);
//     return QMDDGate(plusXEdge);
// }();

// const QMDDGate gate::MINUS_X_GATE = [] {
//     complex<double> minusXWeight = 1 / sqrt(2.0);
//     shared_ptr<QMDDNode> minusXNode = make_shared<QMDDNode>(4);

//     minusXNode->children[0] = QMDDEdge(1, nullptr);
//     minusXNode->children[1] = QMDDEdge(-i, nullptr);
//     minusXNode->children[2] = QMDDEdge(-i, nullptr);
//     minusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusXEdge(minusXWeight, minusXNode);
//     return QMDDGate(minusXEdge);
// }();

QMDDGate gate::Y() {
    return QMDDGate(QMDDEdge(-i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(-1.0, nullptr), QMDDEdge(.0, nullptr)}
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
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))}
    })));
}

QMDDGate gate::CX2() {
    shared_ptr<QMDDNode> cx2Node1, cx2Node2;

    boost::fibers::fiber f1([&cx2Node1]() {
        cx2Node1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        });
    });

    boost::fibers::fiber f2([&cx2Node2]() {
        cx2Node2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
        });
    });

    f1.join();
    f2.join();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, cx2Node1), QMDDEdge(1.0, cx2Node2)},
        {QMDDEdge(1.0, cx2Node2), QMDDEdge(1.0, cx2Node1)}
    })));
}

QMDDGate gate::varCX() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    })));
}

QMDDGate gate::CZ() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::Z().getStartNode()))}
    })));
}

QMDDGate gate::DCNOT() {
    shared_ptr<QMDDNode> dcnotNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> dcnotNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> dcnotNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    shared_ptr<QMDDNode> dcnotNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
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
        return QMDDGate(mathUtils::mul(mathUtils::mul(gate::CX1().getInitialEdge(), gate::CX2().getInitialEdge()), gate::CX1().getInitialEdge()));
    } else {
        shared_ptr<QMDDNode> swapNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        });

        shared_ptr<QMDDNode> swapNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        });

        shared_ptr<QMDDNode> swapNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        });

        shared_ptr<QMDDNode> swapNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
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
    shared_ptr<QMDDNode> iswapNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> iswapNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> iswapNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> iswapNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
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
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::S().getStartNode()))}
    })));
}

QMDDGate gate::Rx(double theta) {
    return QMDDGate(QMDDEdge(cos(theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-i * tan(theta / 2.0), nullptr)},
        {QMDDEdge(-i * tan(theta / 2.0), nullptr), QMDDEdge(1.0, nullptr)}
    })));
}

QMDDGate gate::Ry(double theta) {
    return QMDDGate(QMDDEdge(cos(theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-tan(theta / 2.0), nullptr)},
        {QMDDEdge(tan(theta / 2.0), nullptr), QMDDEdge(1.0, nullptr)}
    })));
}

QMDDGate gate::Rz(double theta) {
    return QMDDGate(QMDDEdge(exp(-i * theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * theta), nullptr)}
    })));
}

QMDDGate gate::Rxx(double phi) {
    return QMDDGate(QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(-i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::X().getStartNode()))},
        {QMDDEdge(-i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::X().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    })));
}

QMDDGate gate::Ryy(double phi) {
    return QMDDGate(QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::Y().getStartNode()))},
        {QMDDEdge(-i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::Y().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    })));
}

QMDDGate gate::Rzz(double phi) {
    return QMDDGate(QMDDEdge(exp(-i * phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::P(phi).getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * phi), make_shared<QMDDNode>(*gate::P(-phi).getStartNode()))}
    })));
}

QMDDGate gate::Rxy(double phi) {
    shared_ptr<QMDDNode> rxyNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(phi / 2.0), nullptr)}
    });

    shared_ptr<QMDDNode> rxyNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> rxyNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> rxyNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0 * mathUtils::sec(phi / 2.0), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, rxyNode1), QMDDEdge(-i * sin(phi / 2.0), rxyNode2)},
        {QMDDEdge(-i * sin(phi / 2.0), rxyNode3), QMDDEdge(cos(phi / 2.0), rxyNode4)}
    })));
}

QMDDGate gate::SquareSWAP() {
    shared_ptr<QMDDNode> squareSWAPNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge((1.0 + i) / 2.0, nullptr)}
    });

    shared_ptr<QMDDNode> squareSWAPNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> squareSWAPNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> squareSWAPNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0 - i, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, squareSWAPNode1), QMDDEdge((1.0 - i) / 2.0, squareSWAPNode2)},
        {QMDDEdge((1.0 - i) / 2.0, squareSWAPNode3), QMDDEdge((1.0 + i) / 2.0, squareSWAPNode4)}
    })));
}

QMDDGate gate::SquareiSWAP() {
    shared_ptr<QMDDNode> squareiSWAPNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0 / sqrt(2.0), nullptr)}
    });

    shared_ptr<QMDDNode> squareiSWAPNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> squareiSWAPNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> squareiSWAPNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(sqrt(2.0), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, squareiSWAPNode1), QMDDEdge(i / sqrt(2.0), squareiSWAPNode2)},
        {QMDDEdge(i / sqrt(2.0), squareiSWAPNode3), QMDDEdge(1.0 / sqrt(2.0), squareiSWAPNode4)}
    })));
}

QMDDGate gate::SWAPalpha(double alpha) {
    shared_ptr<QMDDNode> SWAPalphaNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, nullptr)}
    });

    shared_ptr<QMDDNode> SWAPalphaNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> SWAPalphaNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> SWAPalphaNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(2.0 / (1.0 + exp(i * M_PI * alpha)), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, SWAPalphaNode1), QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode2)},
        {QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode3), QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode4)}
    })));
}

QMDDGate gate::FREDKIN() {
    shared_ptr<QMDDNode> fredkinNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    // shared_ptr<QMDDNode> fredkinNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    // });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fredkinNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::SWAP().getStartNode()))}
    })));
}

QMDDGate gate::U(double theta, double phi, double lambda) {
    return QMDDGate(QMDDEdge(cos(theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-exp(i * lambda) * tan(theta / 2.0), nullptr)},
        {QMDDEdge(exp(i * phi) * tan(theta / 2.0), nullptr), QMDDEdge(exp(i * (lambda + phi)), nullptr)}
    })));
}

QMDDGate gate::BARENCO(double alpha, double phi, double theta) {
    shared_ptr<QMDDNode> barencoNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-i * exp(-i * phi) * tan(theta), nullptr)},
        {QMDDEdge(-i * exp(i * phi) * tan(theta), nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * alpha) * cos(theta), barencoNode1)}
    })));
}

QMDDGate gate::B() {
    shared_ptr<QMDDNode> bNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(3.0 * M_PI / 8.0) * mathUtils::sec(M_PI / 8.0), nullptr)}
    });

    shared_ptr<QMDDNode> bNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(sin(3.0 * M_PI / 8.0) * mathUtils::csc(M_PI / 8.0), nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> bNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(sin(M_PI / 8.0) * mathUtils::csc(3.0 * M_PI / 8.0), nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> bNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(M_PI / 8.0) * mathUtils::sec(3.0 * M_PI / 8.0), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(M_PI / 8.0), bNode1), QMDDEdge(i * sin(M_PI / 8.0), bNode2)},
        {QMDDEdge(i * sin(3.0 * M_PI / 8.0), bNode3), QMDDEdge(cos(3.0 * M_PI / 8.0), bNode4)}
    })));
}

QMDDGate gate::CSX() {
    shared_ptr<QMDDNode> csxNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(exp(-i * M_PI / 2.0), nullptr)},
        {QMDDEdge(exp(-i * M_PI / 2.0), nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * M_PI / 4.0), csxNode1)}
    })));
}

QMDDGate gate::N(double a, double b, double c) {
    shared_ptr<QMDDNode> nNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(-2.0 * i * c) * cos(a + b) * mathUtils::sec(a - b), nullptr)}
    });

    shared_ptr<QMDDNode> nNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(exp(-2.0 * i * c) * sin(a + b) * mathUtils::csc(a - b), nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> nNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(exp(2.0 * i * c) * sin(a - b) * mathUtils::csc(a + b), nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> nNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(2.0 * i * c) * cos(a - b) * mathUtils::sec(a + b), nullptr)}
    });

    return QMDDGate(QMDDEdge(exp(i * c) * cos(a - b), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nNode1), QMDDEdge(i * tan(a - b), nNode2)},
        {QMDDEdge(i * exp(-2.0 * i * c) * sin(a + b) * mathUtils::sec(a - b), nNode3), QMDDEdge(exp(-2.0 * i * c) * cos(a + b) * mathUtils::sec(a - b), nNode4)}
    })));
}

QMDDGate gate::DB() {
    shared_ptr<QMDDNode> dbNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(3.0 * M_PI / 8.0), nullptr)}
    });

    shared_ptr<QMDDNode> dbNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> dbNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> dbNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0 * mathUtils::sec(3.0 * M_PI / 8.0), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dbNode1), QMDDEdge(-i * sin(3.0 * M_PI / 8.0), dbNode2)},
        {QMDDEdge(-i * sin(3.0 * M_PI / 8.0), dbNode3), QMDDEdge(cos(3.0 * M_PI / 8.0), dbNode4)}
    })));
}

QMDDGate gate::ECR() {
    shared_ptr<QMDDNode> ecrNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
        {QMDDEdge(i, nullptr), QMDDEdge(1.0, nullptr)}
    });

    shared_ptr<QMDDNode> ecrNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)},
        {QMDDEdge(-i, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, ecrNode1)},
        {QMDDEdge(1.0, ecrNode2), QMDDEdge(.0, nullptr)}
    })));
}

QMDDGate gate::fSim(double theta, double phi) {
    shared_ptr<QMDDNode> fSimNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(theta), nullptr)}
    });

    shared_ptr<QMDDNode> fSimNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> fSimNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> fSimNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * phi) * mathUtils::sec(theta), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fSimNode1), QMDDEdge(-i * sin(theta), fSimNode2)},
        {QMDDEdge(-i * sin(theta), fSimNode3), QMDDEdge(cos(theta), fSimNode4)}
    })));
}

QMDDGate gate::G(double theta) {
    shared_ptr<QMDDNode> gNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(cos(theta), nullptr)}
    });

    shared_ptr<QMDDNode> gNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> gNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> gNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0 * mathUtils::sec(theta), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, gNode1), QMDDEdge(-sin(theta), gNode2)},
        {QMDDEdge(sin(theta), gNode3), QMDDEdge(cos(theta), gNode4)}
    })));
}

QMDDGate gate::M() {
    shared_ptr<QMDDNode> mNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> mNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)}
    });

    shared_ptr<QMDDNode> mNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)}
    });

    shared_ptr<QMDDNode> mNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, mNode1), QMDDEdge(i, mNode2)},
        {QMDDEdge(1.0, mNode3), QMDDEdge(i, mNode4)}
    })));
}

QMDDGate gate::syc() {
    shared_ptr<QMDDNode> sycNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> sycNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> sycNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> sycNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, sycNode1), QMDDEdge(-i, sycNode2)},
        {QMDDEdge(-i, sycNode3), QMDDEdge(exp(-i * M_PI / 6.0), sycNode4)}
    })));
}

QMDDGate gate::CZS(double theta, double phi, double gamma) {
    shared_ptr<QMDDNode> czsNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(-exp(i * gamma) * std::pow(sin(theta / 2.0), 2) + std::pow(cos(theta / 2.0), 2), nullptr)}
    });

    shared_ptr<QMDDNode> czsNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> czsNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> czsNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(-exp(i * gamma) / (-exp(i * gamma) * std::pow(cos(theta / 2.0), 2) + std::pow(sin(theta / 2.0), 2)), nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, czsNode1), QMDDEdge((1.0 + exp(i * gamma)) / 2.0 * exp(-i * phi) * sin(theta), czsNode2)},
        {QMDDEdge((1.0 + exp(i * gamma)) / 2.0 * exp(i * phi) * sin(theta), czsNode3), QMDDEdge(-exp(i * gamma) * std::pow(cos(theta / 2.0), 2) + std::pow(sin(theta / 2.0), 2), czsNode4)}
    })));
}

QMDDGate gate::D(double theta) {
    shared_ptr<QMDDNode> dNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    // shared_ptr<QMDDNode> dNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    // });

    shared_ptr<QMDDNode> dNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-i * tan(theta), nullptr)},
        {QMDDEdge(-i * tan(theta), nullptr), QMDDEdge(1.0, nullptr)}
    });

    shared_ptr<QMDDNode> dNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(i * cos(theta), dNode3)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, dNode4)}
    })));
}

QMDDGate gate::RCCX() {
    shared_ptr<QMDDNode> rccxNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    // shared_ptr<QMDDNode> rccxNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    // });

    shared_ptr<QMDDNode> rccxNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::Z().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, rccxNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, rccxNode3)}
    })));
}

QMDDGate gate::PG() {
    shared_ptr<QMDDNode> pgNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    // shared_ptr<QMDDNode> pgNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    // });

    shared_ptr<QMDDNode> pgNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))},
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, pgNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, pgNode3)}
    })));
}

QMDDGate gate::Toff() {
    shared_ptr<QMDDNode> toffNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    // shared_ptr<QMDDNode> toffNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    // });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, toffNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::CX1().getStartNode()))}
    })));
}

QMDDGate gate::fFredkin() {
    shared_ptr<QMDDNode> fFredkinNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    // shared_ptr<QMDDNode> fFredkinNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
    //     {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))}
    // });

    shared_ptr<QMDDNode> fFredkinNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> fFredkinNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> fFredkinNode5 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    });

    shared_ptr<QMDDNode> fFredkinNode6 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    });

    shared_ptr<QMDDNode> fFredkinNode7 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fFredkinNode3), QMDDEdge(1.0, fFredkinNode4)},
        {QMDDEdge(1.0, fFredkinNode5), QMDDEdge(-1.0, fFredkinNode6)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fFredkinNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, fFredkinNode7)}
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