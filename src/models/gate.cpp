#include "gate.hpp"

#include <iostream>

using namespace std;

complex<double> i(0.0, 1.0);

QMDDGate gate::ZERO() {
    return QMDDGate(QMDDEdge(.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr)
    })));
};


QMDDGate gate::I() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(1.0, nullptr)
    })));
};

QMDDGate gate::Ph(double delta) {
    return QMDDGate(QMDDEdge(exp(i * delta), make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(1.0, nullptr)
    })));
}

QMDDGate gate::X() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr)
    })));
};


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
    return QMDDGate(QMDDEdge(i, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(.0, nullptr),
        QMDDEdge(-1.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr)
    })));
};

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
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(-1.0, nullptr)
    })));
};

QMDDGate gate::S() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(i, nullptr)
    })));
};

QMDDGate gate::V() {
    return QMDDGate(QMDDEdge(1.0 / 2.0 + i / 2.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(i, nullptr),
        QMDDEdge(i, nullptr),
        QMDDEdge(1.0, nullptr)
    })));
};

QMDDGate gate::H() {
    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(-1.0, nullptr)
    })));
};

QMDDGate gate::CX1() {
    auto iNode = make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(1.0, nullptr)
    });

    auto zeroNode = make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr)
    });

    auto xNode = make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr)
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, iNode),
        QMDDEdge(.0, zeroNode),
        QMDDEdge(.0, zeroNode),
        QMDDEdge(1.0, xNode)
    })));
};

QMDDGate gate::CX2() {

    auto cx2Node1 = make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr)
    });

    auto cx2Node2 = make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(1.0, nullptr)
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, cx2Node1),
        QMDDEdge(1.0, cx2Node2),
        QMDDEdge(1.0, cx2Node2),
        QMDDEdge(1.0, cx2Node1)
    })));
}

QMDDGate gate::varCX() {
    complex<double> varCXWeight = 1.0;

    vector<QMDDEdge> varCXChildren = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::X().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()))
    };

    auto varCXNode = make_shared<QMDDNode>(varCXChildren);

    QMDDEdge varCXEdge(varCXWeight, varCXNode);
    return QMDDGate(varCXEdge);
}

QMDDGate gate::CZ() {
    complex<double> czWeight = 1.0;

    vector<QMDDEdge> czChildren = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::Z().getStartNode()))
    };

    auto czNode = make_shared<QMDDNode>(czChildren);

    QMDDEdge czEdge(czWeight, czNode);
    return QMDDGate(czEdge);
}

QMDDGate gate::DCNOT() {
    complex<double> dcnotWeight = 1.0;

    vector<QMDDEdge> dcnotChildren1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto dcnotNode1 = make_shared<QMDDNode>(dcnotChildren1);

    vector<QMDDEdge> dcnotChildren2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto dcnotNode2 = make_shared<QMDDNode>(dcnotChildren2);

    vector<QMDDEdge> dcnotChildren3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto dcnotNode3 = make_shared<QMDDNode>(dcnotChildren3);

    vector<QMDDEdge> dcnotChildren4 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto dcnotNode4 = make_shared<QMDDNode>(dcnotChildren4);


    vector<QMDDEdge> dcnotChildren = {
        QMDDEdge(1.0, dcnotNode1),
        QMDDEdge(1.0, dcnotNode2),
        QMDDEdge(1.0, dcnotNode3),
        QMDDEdge(1.0, dcnotNode4)
    };

    auto dcnotNode = make_shared<QMDDNode>(dcnotChildren);

    QMDDEdge dcnotEdge(dcnotWeight, dcnotNode);
    return QMDDGate(dcnotEdge);
}

QMDDGate gate::SWAP() {
    complex<double> swapWeight = 1.0;

    vector<QMDDEdge> swapChildren1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
    };

    auto swapNode1 = make_shared<QMDDNode>(swapChildren1);

    vector<QMDDEdge> swapChildren2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto swapNode2 = make_shared<QMDDNode>(swapChildren2);

    vector<QMDDEdge> swapChildren3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto swapNode3 = make_shared<QMDDNode>(swapChildren3);

    vector<QMDDEdge> swapChildren4 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto swapNode4 = make_shared<QMDDNode>(swapChildren4);

    vector<QMDDEdge> swapChildren = {
        QMDDEdge(1.0, swapNode1),
        QMDDEdge(1.0, swapNode2),
        QMDDEdge(1.0, swapNode3),
        QMDDEdge(1.0, swapNode4)
    };

    auto swapNode = make_shared<QMDDNode>(swapChildren);

    QMDDEdge swapEdge(swapWeight, swapNode);
    return QMDDGate(swapEdge);
}

QMDDGate gate::iSWAP() {
    complex<double> iswapWeight = 1.0;

    vector<QMDDEdge> iswapChildren1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
    };

    auto iswapNode1 = make_shared<QMDDNode>(iswapChildren1);

    vector<QMDDEdge> iswapChildren2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto iswapNode2 = make_shared<QMDDNode>(iswapChildren2);

    vector<QMDDEdge> iswapChildren3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto iswapNode3 = make_shared<QMDDNode>(iswapChildren3);

    vector<QMDDEdge> iswapChildren4 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto iswapNode4 = make_shared<QMDDNode>(iswapChildren4);

    vector<QMDDEdge> iswapChildren = {
        QMDDEdge(1.0, iswapNode1),
        QMDDEdge(i, iswapNode2),
        QMDDEdge(i, iswapNode3),
        QMDDEdge(1.0, iswapNode4)
    };

    auto iswapNode = make_shared<QMDDNode>(iswapChildren);

    QMDDEdge iswapEdge(iswapWeight, iswapNode);
    return QMDDGate(iswapEdge);
}

QMDDGate gate::P(double phi) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(exp(i * phi), nullptr)
    })));
}

QMDDGate gate::T() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(exp(i * M_PI / 4.0), nullptr)
    })));
};

QMDDGate gate::CP(double phi) {
    complex<double> cpWeight = 1.0;

    vector<QMDDEdge> cpChildren = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::P(phi).getStartNode()))
    };

    auto cpNode = make_shared<QMDDNode>(cpChildren);

    QMDDEdge cpEdge(cpWeight, cpNode);
    return QMDDGate(cpEdge);
}

QMDDGate gate::CS() {
    complex<double> csWeight = 1.0;

    vector<QMDDEdge> csChildren = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::S().getStartNode()))
    };

    auto csNode = make_shared<QMDDNode>(csChildren);

    QMDDEdge csEdge(csWeight, csNode);
    return QMDDGate(csEdge);
}

QMDDGate gate::Rx(double theta) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(cos(theta / 2.0), nullptr),
        QMDDEdge(-i * sin(theta / 2.0), nullptr),
        QMDDEdge(-i * sin(theta / 2.0), nullptr),
        QMDDEdge(cos(theta / 2.0), nullptr)
    })));
}

QMDDGate gate::Ry(double theta) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(cos(theta / 2.0), nullptr),
        QMDDEdge(-sin(theta / 2.0), nullptr),
        QMDDEdge(sin(theta / 2.0), nullptr),
        QMDDEdge(cos(theta / 2.0), nullptr)
    })));
}

QMDDGate gate::Rz(double theta) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(exp(-i * theta / 2.0), nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(.0, nullptr),
        QMDDEdge(exp(i * theta / 2.0), nullptr)
    })));
}

QMDDGate gate::Rxx(double phi) {
    complex<double> rxxWeight = 1.0;

    vector<QMDDEdge> rxxChildren = {
        QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(-i * sin(phi / 2.0), shared_ptr<QMDDNode>(gate::X().getStartNode())),
        QMDDEdge(-i * sin(phi / 2.0), shared_ptr<QMDDNode>(gate::X().getStartNode())),
        QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode()))
    };

    auto rxxNode = make_shared<QMDDNode>(rxxChildren);

    QMDDEdge rxxEdge(rxxWeight, rxxNode);
    return QMDDGate(rxxEdge);
}

QMDDGate gate::Ryy(double phi) {
    complex<double> ryyWeight = 1.0;

    vector<QMDDEdge> ryyChildren = {
        QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(-sin(phi / 2.0), shared_ptr<QMDDNode>(gate::Y().getStartNode())),
        QMDDEdge(sin(phi / 2.0), shared_ptr<QMDDNode>(gate::Y().getStartNode())),
        QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode()))
    };

    auto ryyNode = make_shared<QMDDNode>(ryyChildren);

    QMDDEdge ryyEdge(ryyWeight, ryyNode);
    return QMDDGate(ryyEdge);
}

QMDDGate gate::Rzz(double phi) {
    complex<double> rzzWeight = 1.0;

    vector<QMDDEdge> rzzChildren = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::Rz(phi).getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(exp(i * phi / 2.0), shared_ptr<QMDDNode>(gate::P(-phi).getStartNode()))
    };

    auto rzzNode = make_shared<QMDDNode>(rzzChildren);

    QMDDEdge rzzEdge(rzzWeight, rzzNode);
    return QMDDGate(rzzEdge);
}

QMDDGate gate::Rxy(double phi) {
    complex<double> rxyWeight = 1.0;

    vector<QMDDEdge> rxyChildren1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(cos(phi / 2.0), nullptr)
    };

    auto rxyNode1 = make_shared<QMDDNode>(rxyChildren1);

    vector<QMDDEdge> rxyChildren2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto rxyNode2 = make_shared<QMDDNode>(rxyChildren2);

    vector<QMDDEdge> rxyChildren3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto rxyNode3 = make_shared<QMDDNode>(rxyChildren3);

    vector<QMDDEdge> rxyChildren4 = {
        QMDDEdge(cos(phi / 2.0), nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto rxyNode4 = make_shared<QMDDNode>(rxyChildren4);

    vector<QMDDEdge> rxyChildren = {
        QMDDEdge(1.0, rxyNode1),
        QMDDEdge(-i * sin(phi / 2.0), rxyNode2),
        QMDDEdge(-i * sin(phi / 2.0), rxyNode3),
        QMDDEdge(1.0, rxyNode4)
    };

    auto rxyNode = make_shared<QMDDNode>(rxyChildren);

    QMDDEdge rxyEdge(rxyWeight, rxyNode);
    return QMDDGate(rxyEdge);
}

QMDDGate gate::SquareSWAP() {
    complex<double> squareSWAPWeight = 1.0;

    vector<QMDDEdge> squareSWAPChildren1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge((1.0 + i) / 2.0, nullptr)
    };

    auto squareSWAPNode1 = make_shared<QMDDNode>(squareSWAPChildren1);

    vector<QMDDEdge> squareSWAPChildren2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto squareSWAPNode2 = make_shared<QMDDNode>(squareSWAPChildren2);

    vector<QMDDEdge> squareSWAPChildren3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto squareSWAPNode3 = make_shared<QMDDNode>(squareSWAPChildren3);

    vector<QMDDEdge> squareSWAPChildren4 = {
        QMDDEdge((1.0 + i) / 2.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto squareSWAPNode4 = make_shared<QMDDNode>(squareSWAPChildren4);

    vector<QMDDEdge> squareSWAPChildren = {
        QMDDEdge(1.0, squareSWAPNode1),
        QMDDEdge((1.0 - i) / 2.0, squareSWAPNode2),
        QMDDEdge((1.0 - i) / 2.0, squareSWAPNode3),
        QMDDEdge(1.0, squareSWAPNode4)
    };

    auto squareSWAPNode = make_shared<QMDDNode>(squareSWAPChildren);

    QMDDEdge squareSWAPEdge(squareSWAPWeight, squareSWAPNode);
    return QMDDGate(squareSWAPEdge);
}

QMDDGate gate::SquareiSWAP() {
    complex<double> squareiSWAPWeight = 1.0;

    vector<QMDDEdge> squareiSWAPChildren1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0 / sqrt(2.0), nullptr)
    };

    auto squareiSWAPNode1 = make_shared<QMDDNode>(squareiSWAPChildren1);

    vector<QMDDEdge> squareiSWAPChildren2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto squareiSWAPNode2 = make_shared<QMDDNode>(squareiSWAPChildren2);

    vector<QMDDEdge> squareiSWAPChildren3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto squareiSWAPNode3 = make_shared<QMDDNode>(squareiSWAPChildren3);

    vector<QMDDEdge> squareiSWAPChildren4 = {
        QMDDEdge(1.0 / sqrt(2.0), nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto squareiSWAPNode4 = make_shared<QMDDNode>(squareiSWAPChildren4);

    vector<QMDDEdge> squareiSWAPChildren = {
        QMDDEdge(1.0, squareiSWAPNode1),
        QMDDEdge(i / sqrt(2.0), squareiSWAPNode2),
        QMDDEdge(i / sqrt(2.0), squareiSWAPNode3),
        QMDDEdge(1.0, squareiSWAPNode4)
    };

    auto squareiSWAPNode = make_shared<QMDDNode>(squareiSWAPChildren);

    QMDDEdge squareiSWAPEdge(squareiSWAPWeight, squareiSWAPNode);
    return QMDDGate(squareiSWAPEdge);
}

QMDDGate gate::SWAPalpha(double alpha) {
    complex<double> SWAPalphaWeight = 1.0;

    vector<QMDDEdge> SWAPalphaChildren1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, nullptr)
    };

    auto SWAPalphaNode1 = make_shared<QMDDNode>(SWAPalphaChildren1);

    vector<QMDDEdge> SWAPalphaChildren2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto SWAPalphaNode2 = make_shared<QMDDNode>(SWAPalphaChildren2);

    vector<QMDDEdge> SWAPalphaChildren3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto SWAPalphaNode3 = make_shared<QMDDNode>(SWAPalphaChildren3);

    vector<QMDDEdge> SWAPalphaChildren4 = {
        QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto SWAPalphaNode4 = make_shared<QMDDNode>(SWAPalphaChildren4);

    vector<QMDDEdge> SWAPalphaChildren = {
        QMDDEdge(1.0, SWAPalphaNode1),
        QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode2),
        QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode3),
        QMDDEdge(1.0, SWAPalphaNode4)
    };

    auto SWAPalphaNode = make_shared<QMDDNode>(SWAPalphaChildren);

    QMDDEdge SWAPalphaEdge(SWAPalphaWeight, SWAPalphaNode);
    return QMDDGate(SWAPalphaEdge);
}

QMDDGate gate::FREDKIN() {
    complex<double> fredkinWeight = 1.0;

    vector<QMDDEdge> fredkinChildren1 = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()))
    };

    auto fredkinNode1 = make_shared<QMDDNode>(fredkinChildren1);

    vector<QMDDEdge> fredkinChildren2 = {
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()))
    };

    auto fredkinNode2 = make_shared<QMDDNode>(fredkinChildren2);

    vector<QMDDEdge> fredkinChildren = {
        QMDDEdge(1.0, fredkinNode1),
        QMDDEdge(0.0, fredkinNode2),
        QMDDEdge(0.0, fredkinNode2),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::SWAP().getStartNode()))
    };

    auto fredkinNode = make_shared<QMDDNode>(fredkinChildren);
    QMDDEdge fredkinEdge(fredkinWeight, fredkinNode);
    return QMDDGate(fredkinEdge);
}

QMDDGate gate::U(double theta, double phi, double lambda) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(cos(theta / 2.0), nullptr),
        QMDDEdge(-exp(i * lambda) * sin(theta / 2.0), nullptr),
        QMDDEdge(exp(i * phi) * sin(theta / 2.0), nullptr),
        QMDDEdge(exp(i * (lambda + phi)) * cos(theta / 2.0), nullptr)
    })));
}

QMDDGate gate::BARENCO(double alpha, double phi, double theta) {
    complex<double> barencoWeight = 1.0;

    vector<QMDDEdge> barencoChildren1 = {
        QMDDEdge(exp(i * alpha) * cos(theta), nullptr),
        QMDDEdge(-i * exp(i * (alpha - phi)) * sin(theta), nullptr),
        QMDDEdge(-i * exp(i * (alpha + phi)) * sin(theta), nullptr),
        QMDDEdge(exp(i * alpha) * cos(theta), nullptr)
    };

    auto barencoNode1 = make_shared<QMDDNode>(barencoChildren1);

    vector<QMDDEdge> barencoChildren = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, barencoNode1)
    };

    auto barencoNode = make_shared<QMDDNode>(barencoChildren);

    QMDDEdge barencoEdge(barencoWeight, barencoNode);
    return QMDDGate(barencoEdge);
}

QMDDGate gate::B() {
    complex<double> bWeight = 1.0;

    vector<QMDDEdge> bChildren1 = {
        QMDDEdge(cos(M_PI / 8.0), nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(cos(3.0 * M_PI / 8.0), nullptr)
    };

    auto bNode1 = make_shared<QMDDNode>(bChildren1);

    vector<QMDDEdge> bChildren2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(sin(M_PI / 8.0), nullptr),
        QMDDEdge(sin(3.0 * M_PI / 8.0), nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto bNode2 = make_shared<QMDDNode>(bChildren2);

    vector<QMDDEdge> bChildren = {
        QMDDEdge(1.0, bNode1),
        QMDDEdge(i, bNode2),
        QMDDEdge(i * sin(M_PI / 8.0), shared_ptr<QMDDNode>(gate::X().getStartNode())),
        QMDDEdge(cos(M_PI / 8.0), shared_ptr<QMDDNode>(gate::I().getStartNode()))
    };

    auto bNode = make_shared<QMDDNode>(bChildren);

    QMDDEdge bEdge(bWeight, bNode);
    return QMDDGate(bEdge);
}

QMDDGate gate::N(double a, double b, double c) {
    complex<double> nWeight = 1.0;

    vector<QMDDEdge> nChildren1 = {
        QMDDEdge(exp(i * c) * cos(a - b), nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(exp(-i * c) * cos(a + b), nullptr)
    };

    auto nNode1 = make_shared<QMDDNode>(nChildren1);

    vector<QMDDEdge> nChildren2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(exp(i * c) * sin(a - b), nullptr),
        QMDDEdge(exp(-i * c) * sin(a + b), nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto nNode2 = make_shared<QMDDNode>(nChildren2);

    vector<QMDDEdge> nChildren3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(exp(-i * c) * sin(a + b), nullptr),
        QMDDEdge(exp(i * c) * sin(a - b), nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto nNode3 = make_shared<QMDDNode>(nChildren3);

    vector<QMDDEdge> nChildren4 = {
        QMDDEdge(exp(-i * c) * cos(a + b), nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(exp(i * c) * cos(a - b), nullptr)
    };

    auto nNode4 = make_shared<QMDDNode>(nChildren4);

    vector<QMDDEdge> nChildren = {
        QMDDEdge(1.0, nNode1),
        QMDDEdge(i, nNode2),
        QMDDEdge(i, nNode3),
        QMDDEdge(1.0, nNode4)
    };

    auto nNode = make_shared<QMDDNode>(nChildren);

    QMDDEdge nEdge(nWeight, nNode);
    return QMDDGate(nEdge);
}



// QMDDGate createSDaggerGate() {
//     complex<double> sDaggerWeight = 1.0;
//     auto sDaggerNode = make_shared<QMDDNode>(4);

//     sDaggerNode->children[0] = QMDDEdge(1, nullptr);
//     sDaggerNode->children[1] = QMDDEdge(0, nullptr);
//     sDaggerNode->children[2] = QMDDEdge(0, nullptr);
//     sDaggerNode->children[3] = QMDDEdge(-i, nullptr);

//     QMDDEdge sDaggerEdge(sDaggerWeight, sDaggerNode);
//     return QMDDGate(sDaggerEdge);
// }

// QMDDGate createTDaggerGate() {
//     complex<double> tDaggerWeight = 1.0;
//     QMDDNode* tDaggerNode = new QMDDNode(4);

//     tDaggerNode->children[0] = QMDDEdge(1, nullptr);
//     tDaggerNode->children[1] = QMDDEdge(0, nullptr);
//     tDaggerNode->children[2] = QMDDEdge(0, nullptr);
//     tDaggerNode->children[3] = QMDDEdge(-exp(i * complex<double>(M_PI / 4)), nullptr);

//     QMDDEdge tDaggerEdge(tDaggerWeight, tDaggerNode);
//     return QMDDGate(tDaggerEdge);
// }


// const matrix TOFFOLI_GATE = matrix{ 
//     {1, 0, 0, 0, 0, 0, 0, 0},
//     {0, 1, 0, 0, 0, 0, 0, 0},
//     {0, 0, 1, 0, 0, 0, 0, 0},
//     {0, 0, 0, 1, 0, 0, 0, 0},
//     {0, 0, 0, 0, 1, 0, 0, 0},
//     {0, 0, 0, 0, 0, 1, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 1},
//     {0, 0, 0, 0, 0, 0, 1, 0}
// };




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



// PYBIND11_MODULE(gate_py, m) {
//     py::class_<QMDDNode, std::shared_ptr<QMDDNode>>(m, "QMDDNode")
//         .def(py::init<int>());

//     py::class_<QMDDEdge>(m, "QMDDEdge")
//         .def(py::init<std::complex<double>, std::shared_ptr<QMDDNode>>());

//     py::class_<QMDDGate>(m, "QMDDGate")
//         .def_static("get_h_gate", []() { return gate::H_GATE; }, py::return_value_policy::reference);

//     m.attr("H_GATE") = gate::H_GATE;
// }