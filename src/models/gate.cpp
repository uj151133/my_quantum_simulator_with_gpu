#include "gate.hpp"

#include <iostream>

namespace py = pybind11;

complex<double> i(0.0, 1.0);

QMDDGate gate::ZERO() {
    complex<double> zeroWeight = 0.0;

    vector<QMDDEdge> zeroEdges = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto zeroNode = make_shared<QMDDNode>(zeroEdges);

    QMDDEdge zeroEdge(zeroWeight, zeroNode);
    return QMDDGate(zeroEdge);
};


QMDDGate gate::I() {
    complex<double> iWeight = 1.0;

    vector<QMDDEdge> iEdges = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto iNode = make_shared<QMDDNode>(iEdges);

    QMDDEdge iEdge(iWeight, iNode);
    return QMDDGate(iEdge);
};

QMDDGate gate::Ph(double delta) {
    complex<double> phWeight = exp(i * delta);

    vector<QMDDEdge> phEdges = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto phNode = make_shared<QMDDNode>(phEdges);

    QMDDEdge phEdge(phWeight, phNode);
    return QMDDGate(phEdge);
}

QMDDGate gate::X() {
    complex<double> xWeight = 1.0;

    vector<QMDDEdge> xEdges = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto xNode = make_shared<QMDDNode>(xEdges);

    QMDDEdge xEdge(xWeight, xNode);
    return QMDDGate(xEdge);
};


// const QMDDGate gate::PLUS_X_GATE = [] {
//     complex<double> plusXWeight = 1 / sqrt(2.0);
//     auto plusXNode = make_shared<QMDDNode>(4);

//     plusXNode->edges[0] = QMDDEdge(1, nullptr);
//     plusXNode->edges[1] = QMDDEdge(i, nullptr);
//     plusXNode->edges[2] = QMDDEdge(i, nullptr);
//     plusXNode->edges[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusXEdge(plusXWeight, plusXNode);
//     return QMDDGate(plusXEdge);
// }();

// const QMDDGate gate::MINUS_X_GATE = [] {
//     complex<double> minusXWeight = 1 / sqrt(2.0);
//     auto minusXNode = make_shared<QMDDNode>(4);

//     minusXNode->edges[0] = QMDDEdge(1, nullptr);
//     minusXNode->edges[1] = QMDDEdge(-i, nullptr);
//     minusXNode->edges[2] = QMDDEdge(-i, nullptr);
//     minusXNode->edges[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusXEdge(minusXWeight, minusXNode);
//     return QMDDGate(minusXEdge);
// }();



QMDDGate gate::Y() {
    complex<double> yWeight = i;
    
    vector<QMDDEdge> yEdges = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(-1.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto yNode = make_shared<QMDDNode>(yEdges);

    QMDDEdge yEdge(yWeight, yNode);
    return QMDDGate(yEdge);
};

// QMDDGate createPlusYGate() {
//     complex<double> plusYWeight = 1 / sqrt(2.0);
//     QMDDNode* plusYNode = new QMDDNode(4);

//     plusYNode->edges[0] = QMDDEdge(1, nullptr);
//     plusYNode->edges[1] = QMDDEdge(1, nullptr);
//     plusYNode->edges[2] = QMDDEdge(-1, nullptr);
//     plusYNode->edges[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusYEdge(plusYWeight, plusYNode);
//     return QMDDGate(plusYEdge);
// }

// QMDDGate createMinusYGate() {
//     complex<double> minusYWeight = 1 / sqrt(2.0);
//     QMDDNode* minusYNode = new QMDDNode(4);

//     minusYNode->edges[0] = QMDDEdge(1, nullptr);
//     minusYNode->edges[1] = QMDDEdge(-1, nullptr);
//     minusYNode->edges[2] = QMDDEdge(1, nullptr);
//     minusYNode->edges[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusYEdge(minusYWeight, minusYNode);
//     return QMDDGate(minusYEdge);
// }

QMDDGate gate::Z() {
    complex<double> zWeight = 1.0;
    
    vector<QMDDEdge> zEdges = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(-1.0, nullptr)
    };

    auto zNode = make_shared<QMDDNode>(zEdges);

    QMDDEdge zEdge(zWeight, zNode);
    return QMDDGate(zEdge);
};

QMDDGate gate::S() {
    complex<double> sWeight = 1.0;
    
    vector<QMDDEdge> sEdges = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(i, nullptr)
    };

    auto sNode = make_shared<QMDDNode>(sEdges);

    QMDDEdge sEdge(sWeight, sNode);
    return QMDDGate(sEdge);
};

QMDDGate gate::V() {
    complex<double> vWeight = 1.0 / 2.0 + i / 2.0;

    vector<QMDDEdge> vEdges = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(i, nullptr),
        QMDDEdge(i, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto vNode = make_shared<QMDDNode>(vEdges);

    QMDDEdge vEdge(vWeight, vNode);
    return QMDDGate(vEdge);
};

QMDDGate gate::H() {
    complex<double> hWeight = 1.0 / sqrt(2.0);

    vector<QMDDEdge> hEdges = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(-1.0, nullptr)
    };

    auto hNode = make_shared<QMDDNode>(hEdges);

    QMDDEdge hEdge(hWeight, hNode);
    return QMDDGate(hEdge);
};

QMDDGate gate::CX1() {
    complex<double> cx1Weight = 1.0;

    vector<QMDDEdge> cx1Edges = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::X().getStartNode()))
    };

    auto cx1Node = make_shared<QMDDNode>(cx1Edges);

    QMDDEdge cx1Edge(cx1Weight, cx1Node);
    return QMDDGate(cx1Edge);
};

QMDDGate gate::CX2() {
    complex<double> cx2Weight = 1.0;

    vector<QMDDEdge> cx2Edges1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto cx2Node1 = make_shared<QMDDNode>(cx1Edges1);

    vector<QMDDEdge> cx2Edges2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto cx2Node2 = make_shared<QMDDNode>(cx2Edges2);

    vector<QMDDEdge> cx2Edges = {
        QMDDEdge(1.0, cx2Node1),
        QMDDEdge(1.0, cx2Node2),
        QMDDEdge(1.0, cx2Node2),
        QMDDEdge(1.0, cx2Node1)
    };

    auto cx2Node = make_shared<QMDDNode>(cx2Edges);

    QMDDEdge cx2Edge(cx2Weight, cx2Node);
    return QMDDGate(cx2Edge);
}

QMDDGate gate::varCX() {
    complex<double> varCXWeight = 1.0;

    vector<QMDDEdge> varCXEdges = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::X().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()))
    };

    auto varCXNode = make_shared<QMDDNode>(varCXEdges);

    QMDDEdge varCXEdge(varCXWeight, varCXNode);
    return QMDDGate(varCXEdge);
}

QMDDGate gate::CZ() {
    complex<double> czWeight = 1.0;

    vector<QMDDEdge> czEdges = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::Z().getStartNode()))
    };

    auto czNode = make_shared<QMDDNode>(czEdges);

    QMDDEdge czEdge(czWeight, czNode);
    return QMDDGate(czEdge);
}

QMDDGate gate::DCNOT() {
    complex<double> dcnotWeight = 1.0;

    vector<QMDDEdge> dcnotEdges1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto dcnotNode1 = make_shared<QMDDNode>(dcnotEdges1);

    vector<QMDDEdge> dcnotEdges2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto dcnotNode2 = make_shared<QMDDNode>(dcnotEdges2);

    vector<QMDDEdge> dcnotEdges3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto dcnotNode3 = make_shared<QMDDNode>(dcnotEdges3);

    vector<QMDDEdge> dcnotEdges4 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto dcnotNode4 = make_shared<QMDDNode>(dcnotEdges4);


    vector<QMDDEdge> dcnotEdges = {
        QMDDEdge(1.0, dcnotNode1),
        QMDDEdge(1.0, dcnotNode2),
        QMDDEdge(1.0, dcnotNode3),
        QMDDEdge(1.0, dcnotNode4)
    };

    auto dcnotNode = make_shared<QMDDNode>(dcnotEdges);

    QMDDEdge dcnotEdge(dcnotWeight, dcnotNode);
    return QMDDGate(dcnotEdge);
}

QMDDGate gate::SWAP() {
    complex<double> swapWeight = 1.0;

    vector<QMDDEdge> swapEdges1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
    };

    auto swapNode1 = make_shared<QMDDNode>(swapEdges1);

    vector<QMDDEdge> swapEdges2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto swapNode2 = make_shared<QMDDNode>(swapEdges2);

    vector<QMDDEdge> swapEdges3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto swapNode3 = make_shared<QMDDNode>(swapEdges3);

    vector<QMDDEdge> swapEdges4 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto swapNode4 = make_shared<QMDDNode>(swapEdges4);

    vector<QMDDEdge> swapEdges = {
        QMDDEdge(1.0, swapNode1),
        QMDDEdge(1.0, swapNode2),
        QMDDEdge(1.0, swapNode3),
        QMDDEdge(1.0, swapNode4)
    };

    auto swapNode = make_shared<QMDDNode>(swapEdges);

    QMDDEdge swapEdge(swapWeight, swapNode);
    return QMDDGate(swapEdge);
}

QMDDGate gate::iSWAP() {
    complex<double> iswapWeight = 1.0;

    vector<QMDDEdge> iswapEdges1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
    };

    auto iswapNode1 = make_shared<QMDDNode>(iswapEdges1);

    vector<QMDDEdge> iswapEdges2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto iswapNode2 = make_shared<QMDDNode>(iswapEdges2);

    vector<QMDDEdge> iswapEdges3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto iswapNode3 = make_shared<QMDDNode>(iswapEdges3);

    vector<QMDDEdge> iswapEdges4 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto iswapNode4 = make_shared<QMDDNode>(iswapEdges4);


    vector<QMDDEdge> iswapEdges = {
        QMDDEdge(1.0, iswapNode1),
        QMDDEdge(i, iswapNode2),
        QMDDEdge(i, iswapNode3),
        QMDDEdge(1.0, iswapNode4)
    };

    auto iswapNode = make_shared<QMDDNode>(iswapEdges);

    QMDDEdge iswapEdge(iswapWeight, iswapNode);
    return QMDDGate(iswapEdge);
}

QMDDGate gate::P(double phi) {
    complex<double> pWeight = 1.0;

    vector<QMDDEdge> pEdges = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(exp(i * phi), nullptr)
    };

    auto pNode = make_shared<QMDDNode>(pEdges);

    QMDDEdge pEdge(pWeight, pNode);
    return QMDDGate(pEdge);
}

QMDDGate gate::T() {
    complex<double> tWeight = 1.0;

    vector<QMDDEdge> tEdges = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(exp((i * M_PI) / 4.0), nullptr)
    };

    auto tNode = make_shared<QMDDNode>(tEdges);

    QMDDEdge tEdge(tWeight, tNode);
    return QMDDGate(tEdge);
};

QMDDGate gate::CP(double phi) {
    complex<double> cpWeight = 1.0;

    vector<QMDDEdge> cpEdges = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::P(phi).getStartNode()))
    };

    auto cpNode = make_shared<QMDDNode>(cpEdges);

    QMDDEdge cpEdge(cpWeight, cpNode);
    return QMDDGate(cpEdge);
}

QMDDGate gate::CS() {
    complex<double> csWeight = 1.0;

    vector<QMDDEdge> csEdges = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::S().getStartNode()))
    };

    auto csNode = make_shared<QMDDNode>(csEdges);

    QMDDEdge csEdge(csWeight, csNode);
    return QMDDGate(csEdge);
}

QMDDGate gate::Rx(double theta) {
    complex<double> rxWeight = 1.0;

    vector<QMDDEdge> rxEdges = {
        QMDDEdge(cos(theta / 2.0), nullptr),
        QMDDEdge(-i * sin(theta / 2.0), nullptr),
        QMDDEdge(-i * sin(theta / 2.0), nullptr),
        QMDDEdge(cos(theta / 2.0), nullptr)
    };

    auto rxNode = make_shared<QMDDNode>(rxEdges);

    QMDDEdge rxEdge(rxWeight, rxNode);
    return QMDDGate(rxEdge);
}

QMDDGate gate::Ry(double theta) {
    complex<double> ryWeight = 1.0;

    vector<QMDDEdge> ryEdges = {
        QMDDEdge(cos(theta / 2.0), nullptr),
        QMDDEdge(-sin(theta / 2.0), nullptr),
        QMDDEdge(sin(theta / 2.0), nullptr),
        QMDDEdge(cos(theta / 2.0), nullptr)
    };

    auto ryNode = make_shared<QMDDNode>(ryEdges);

    QMDDEdge ryEdge(ryWeight, ryNode);
    return QMDDGate(ryEdge);
}

QMDDGate gate::Rz(double theta) {
    complex<double> rzWeight = 1.0;

    vector<QMDDEdge> rzEdges = {
        QMDDEdge(exp(-i * theta / 2.0), nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(exp(i * theta / 2.0), nullptr)
    };

    auto rzNode = make_shared<QMDDNode>(rzEdges);

    QMDDEdge rzEdge(rzWeight, rzNode);
    return QMDDGate(rzEdge);
}

QMDDGate gate::Rxx(double phi) {
    complex<double> rxxWeight = 1.0;

    vector<QMDDEdge> rxxEdges = {
        QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(-i * sin(phi / 2.0), shared_ptr<QMDDNode>(gate::X().getStartNode())),
        QMDDEdge(-i * sin(phi / 2.0), shared_ptr<QMDDNode>(gate::X().getStartNode())),
        QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode()))
    };

    auto rxxNode = make_shared<QMDDNode>(rxxEdges);

    QMDDEdge rxxEdge(rxxWeight, rxxNode);
    return QMDDGate(rxxEdge);
}

QMDDGate gate::Ryy(double phi) {
    complex<double> ryyWeight = 1.0;

    vector<QMDDEdge> ryyEdges = {
        QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode())),
        QMDDEdge(-sin(phi / 2.0), shared_ptr<QMDDNode>(gate::Y().getStartNode())),
        QMDDEdge(sin(phi / 2.0), shared_ptr<QMDDNode>(gate::Y().getStartNode())),
        QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode()))
    };

    auto ryyNode = make_shared<QMDDNode>(ryyEdges);

    QMDDEdge ryyEdge(ryyWeight, ryyNode);
    return QMDDGate(ryyEdge);
}

QMDDGate gate::Rzz(double phi) {
    complex<double> rzzWeight = 1.0;

    vector<QMDDEdge> rzzEdges = {
        QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::Rz(phi).getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode())),
        QMDDEdge(exp(i * phi / 2.0), shared_ptr<QMDDNode>(gate::P(-phi).getStartNode()))
    };

    auto rzzNode = make_shared<QMDDNode>(rzzEdges);

    QMDDEdge rzzEdge(rzzWeight, rzzNode);
    return QMDDGate(rzzEdge);
}

QMDDGate gate::Rxy(double phi) {
    complex<double> rxyWeight = 1.0;

    vector<QMDDEdge> rxyEdges1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(cos(phi / 2.0), nullptr)
    };

    auto rxyNode1 = make_shared<QMDDNode>(rxyEdges1);

    vector<QMDDEdge> rxyEdges2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto rxyNode2 = make_shared<QMDDNode>(rxyEdges2);

    vector<QMDDEdge> rxyEdges3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto rxyNode3 = make_shared<QMDDNode>(rxyEdges3);

    vector<QMDDEdge> rxyEdges4 = {
        QMDDEdge(cos(phi / 2.0), nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto rxyNode4 = make_shared<QMDDNode>(rxyEdges4);

    vector<QMDDEdge> rxyEdges = {
        QMDDEdge(1.0, rxyNode1),
        QMDDEdge(-i * sin(phi / 2.0), rxyNode2),
        QMDDEdge(-i * sin(phi / 2.0), rxyNode3),
        QMDDEdge(1.0, rxyNode4)
    };

    auto rxyNode = make_shared<QMDDNode>(rxyEdges);

    QMDDEdge rxyEdge(rxyWeight, rxyNode);
    return QMDDGate(rxyEdge);
}

QMDDGate gate::SquareSWAP() {
    complex<double> squareSWAPWeight = 1.0;

    vector<QMDDEdge> squareSWAPEdges1 = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge((1.0 + i) / 2.0, nullptr)
    };

    auto squareSWAPNode1 = make_shared<QMDDNode>(squareSWAPEdges1);

    vector<QMDDEdge> squareSWAPEdges2 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto squareSWAPNode2 = make_shared<QMDDNode>(squareSWAPEdges2);

    vector<QMDDEdge> squareSWAPEdges3 = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto squareSWAPNode3 = make_shared<QMDDNode>(squareSWAPEdges3);

    vector<QMDDEdge> squareSWAPEdges4 = {
        QMDDEdge((1.0 + i) / 2.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto squareSWAPNode4 = make_shared<QMDDNode>(squareSWAPEdges4);

    vector<QMDDEdge> squareSWAPEdges = {
        QMDDEdge(1.0, squareSWAPNode1),
        QMDDEdge((1.0 - i) / 2.0, squareSWAPNode2),
        QMDDEdge((1.0 - i) / 2.0, squareSWAPNode3),
        QMDDEdge(1.0, squareSWAPNode4)
    };

    auto squareSWAPNode = make_shared<QMDDNode>(squareSWAPEdges);

    QMDDEdge squareSWAPEdge(squareSWAPWeight, squareSWAPNode);
    return QMDDGate(squareSWAPEdge);
}

QMDDGate gate::SquareiSWAP() {
    complex<double> squareiSWAPWeight = 1.0;
    auto squareiSWAPNode = make_shared<QMDDNode>(4);
    auto squareiSWAPNode1 = make_shared<QMDDNode>(4);
    auto squareiSWAPNode2 = make_shared<QMDDNode>(4);
    auto squareiSWAPNode3 = make_shared<QMDDNode>(4);
    auto squareiSWAPNode4 = make_shared<QMDDNode>(4);

    squareiSWAPNode1->edges[0] = QMDDEdge(1.0, nullptr);
    squareiSWAPNode1->edges[1] = QMDDEdge(0.0, nullptr);
    squareiSWAPNode1->edges[2] = QMDDEdge(0.0, nullptr);
    squareiSWAPNode1->edges[3] = QMDDEdge(1.0 / sqrt(2.0), nullptr);

    squareiSWAPNode2->edges[0] = QMDDEdge(0.0, nullptr);
    squareiSWAPNode2->edges[1] = QMDDEdge(0.0, nullptr);
    squareiSWAPNode2->edges[2] = QMDDEdge(1.0, nullptr);
    squareiSWAPNode2->edges[3] = QMDDEdge(0.0, nullptr);

    squareiSWAPNode3->edges[0] = QMDDEdge(0.0, nullptr);
    squareiSWAPNode3->edges[1] = QMDDEdge(1.0, nullptr);
    squareiSWAPNode3->edges[2] = QMDDEdge(0.0, nullptr);
    squareiSWAPNode3->edges[3] = QMDDEdge(0.0, nullptr);

    squareiSWAPNode4->edges[0] = QMDDEdge(1.0 / sqrt(2.0), nullptr);
    squareiSWAPNode4->edges[1] = QMDDEdge(0.0, nullptr);
    squareiSWAPNode4->edges[2] = QMDDEdge(0.0, nullptr);
    squareiSWAPNode4->edges[3] = QMDDEdge(1.0, nullptr);

    squareiSWAPNode->edges[0] = QMDDEdge(1.0, squareiSWAPNode1);
    squareiSWAPNode->edges[1] = QMDDEdge(i / sqrt(2.0), squareiSWAPNode2);
    squareiSWAPNode->edges[2] = QMDDEdge(i / sqrt(2.0), squareiSWAPNode3);
    squareiSWAPNode->edges[3] = QMDDEdge(1.0, squareiSWAPNode4);

    vector<QMDDEdge> squareiSWAPEdges = {
        QMDDEdge(1.0, squareSWAPNode1),
        QMDDEdge((1.0 - i) / 2.0, squareSWAPNode2),
        QMDDEdge((1.0 - i) / 2.0, squareSWAPNode3),
        QMDDEdge(1.0, squareSWAPNode4)
    };

    auto squareiSWAPNode = make_shared<QMDDNode>(squareiSWAPEdges);

    QMDDEdge squareiSWAPEdge(squareiSWAPWeight, squareiSWAPNode);
    return QMDDGate(squareiSWAPEdge);
}

QMDDGate gate::SWAPalpha(double alpha) {
    complex<double> SWAPalphaWeight = 1.0;
    auto SWAPalphaNode = make_shared<QMDDNode>(4);
    auto SWAPalphaNode1 = make_shared<QMDDNode>(4);
    auto SWAPalphaNode2 = make_shared<QMDDNode>(4);
    auto SWAPalphaNode3 = make_shared<QMDDNode>(4);
    auto SWAPalphaNode4 = make_shared<QMDDNode>(4);

    SWAPalphaNode1->edges[0] = QMDDEdge(1.0, nullptr);
    SWAPalphaNode1->edges[1] = QMDDEdge(0.0, nullptr);
    SWAPalphaNode1->edges[2] = QMDDEdge(0.0, nullptr);
    SWAPalphaNode1->edges[3] = QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, nullptr);

    SWAPalphaNode2->edges[0] = QMDDEdge(0.0, nullptr);
    SWAPalphaNode2->edges[1] = QMDDEdge(0.0, nullptr);
    SWAPalphaNode2->edges[2] = QMDDEdge(1.0, nullptr);
    SWAPalphaNode2->edges[3] = QMDDEdge(0.0, nullptr);

    SWAPalphaNode3->edges[0] = QMDDEdge(0.0, nullptr);
    SWAPalphaNode3->edges[1] = QMDDEdge(1.0, nullptr);
    SWAPalphaNode3->edges[2] = QMDDEdge(0.0, nullptr);
    SWAPalphaNode3->edges[3] = QMDDEdge(0.0, nullptr);

    SWAPalphaNode4->edges[0] = QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, nullptr);
    SWAPalphaNode4->edges[1] = QMDDEdge(0.0, nullptr);
    SWAPalphaNode4->edges[2] = QMDDEdge(0.0, nullptr);
    SWAPalphaNode4->edges[3] = QMDDEdge(1.0, nullptr);

    SWAPalphaNode->edges[0] = QMDDEdge(1.0, SWAPalphaNode1);
    SWAPalphaNode->edges[1] = QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode2);
    SWAPalphaNode->edges[2] = QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode3);
    SWAPalphaNode->edges[3] = QMDDEdge(1.0, SWAPalphaNode4);

    QMDDEdge SWAPalphaEdge(SWAPalphaWeight, SWAPalphaNode);
    return QMDDGate(SWAPalphaEdge);
}

QMDDGate gate::FREDKIN() {
    complex<double> fredkinWeight = 1.0;
    auto fredkinNode = make_shared<QMDDNode>(4);
    auto fredkinNode1 = make_shared<QMDDNode>(4);
    auto fredkinNode2 = make_shared<QMDDNode>(4);

    fredkinNode1->edges[0] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()));
    fredkinNode1->edges[1] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    fredkinNode1->edges[2] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    fredkinNode1->edges[3] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()));

    fredkinNode2->edges[0] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    fredkinNode2->edges[1] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    fredkinNode2->edges[2] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    fredkinNode2->edges[3] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));

    fredkinNode->edges[0] = QMDDEdge(1.0, fredkinNode1);
    fredkinNode->edges[1] = QMDDEdge(0.0, fredkinNode2);
    fredkinNode->edges[2] = QMDDEdge(0.0, fredkinNode2);
    fredkinNode->edges[3] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::SWAP().getStartNode()));

    QMDDEdge fredkinEdge(fredkinWeight, fredkinNode);
    return QMDDGate(fredkinEdge);
}

QMDDGate gate::U(double theta, double phi, double lambda) {
    complex<double> uWeight = 1.0;
    auto uNode = make_shared<QMDDNode>(4);

    uNode->edges[0] = QMDDEdge(cos(theta / 2.0), nullptr);
    uNode->edges[1] = QMDDEdge(-exp(i * lambda) * sin(theta / 2.0), nullptr);
    uNode->edges[2] = QMDDEdge(exp(i * phi) * sin(theta / 2.0), nullptr);
    uNode->edges[3] = QMDDEdge(exp(i * (phi + lambda)) * cos(theta / 2.0), nullptr);

    QMDDEdge uEdge(uWeight, uNode);
    return QMDDGate(uEdge);
}

QMDDGate gate::BARENCO(double alpha, double phi, double theta) {
    complex<double> barencoWeight = 1.0;
    auto barencoNode = make_shared<QMDDNode>(4);
    auto barencoNode1 = make_shared<QMDDNode>(4);

    barencoNode1->edges[0] = QMDDEdge(exp(i * alpha) * cos(theta), nullptr);
    barencoNode1->edges[1] = QMDDEdge(-i * exp(i * (alpha - phi)) * sin(theta), nullptr);
    barencoNode1->edges[2] = QMDDEdge(-i * exp(i * (alpha + phi)) * sin(theta), nullptr);
    barencoNode1->edges[3] = QMDDEdge(exp(i * alpha) * cos(theta), nullptr);

    barencoNode->edges[0] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()));
    barencoNode->edges[1] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    barencoNode->edges[2] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    barencoNode->edges[3] = QMDDEdge(1.0, barencoNode1);

    QMDDEdge barencoEdge(barencoWeight, barencoNode);
    return QMDDGate(barencoEdge);
}

QMDDGate gate::B() {
    complex<double> bWeight = 1.0;
    auto bNode = make_shared<QMDDNode>(4);
    auto bNode1 = make_shared<QMDDNode>(4);
    auto bNode2 = make_shared<QMDDNode>(4);

    bNode1->edges[0] = QMDDEdge(cos(M_PI / 8.0), nullptr);
    bNode1->edges[1] = QMDDEdge(0.0, nullptr);
    bNode1->edges[2] = QMDDEdge(0.0, nullptr);
    bNode1->edges[3] = QMDDEdge(cos(3.0 * M_PI / 8.0), nullptr);

    bNode2->edges[0] = QMDDEdge(0.0, nullptr);
    bNode2->edges[1] = QMDDEdge(sin(M_PI / 8.0), nullptr);
    bNode2->edges[2] = QMDDEdge(sin(3.0 * M_PI / 8.0), nullptr);
    bNode2->edges[3] = QMDDEdge(0.0, nullptr);

    bNode->edges[0] = QMDDEdge(1.0, bNode1);
    bNode->edges[1] = QMDDEdge(i, bNode2);
    bNode->edges[2] = QMDDEdge(i * sin(M_PI / 8.0), shared_ptr<QMDDNode>(gate::X().getStartNode()));
    bNode->edges[3] = QMDDEdge(cos(M_PI / 8.0), shared_ptr<QMDDNode>(gate::I().getStartNode()));

    QMDDEdge bEdge(bWeight, bNode);
    return QMDDGate(bEdge);
}

QMDDGate gate::N(double a, double b, double c) {
    complex<double> nWeight = 1.0;
    auto nNode = make_shared<QMDDNode>(4);
    auto nNode1 = make_shared<QMDDNode>(4);
    auto nNode2 = make_shared<QMDDNode>(4);
    auto nNode3 = make_shared<QMDDNode>(4);
    auto nNode4 = make_shared<QMDDNode>(4);

    nNode1->edges[0] = QMDDEdge(exp(i * c) * cos(a - b), nullptr);
    nNode1->edges[1] = QMDDEdge(0.0, nullptr);
    nNode1->edges[2] = QMDDEdge(0.0, nullptr);
    nNode1->edges[3] = QMDDEdge(exp(-i * c) * cos(a + b), nullptr);

    nNode2->edges[0] = QMDDEdge(0.0, nullptr);
    nNode2->edges[1] = QMDDEdge(exp(i * c) * sin(a - b), nullptr);
    nNode2->edges[2] = QMDDEdge(exp(-i * c) * sin(a + b), nullptr);
    nNode2->edges[3] = QMDDEdge(0.0, nullptr);

    nNode3->edges[0] = QMDDEdge(0.0, nullptr);
    nNode3->edges[1] = QMDDEdge(exp(-i * c) * sin(a + b), nullptr);
    nNode3->edges[2] = QMDDEdge(exp(i * c) * sin(a - b), nullptr);
    nNode3->edges[3] = QMDDEdge(0.0, nullptr);

    nNode4->edges[0] = QMDDEdge(exp(-i * c) * cos(a + b), nullptr);
    nNode4->edges[1] = QMDDEdge(0.0, nullptr);
    nNode4->edges[2] = QMDDEdge(0.0, nullptr);
    nNode4->edges[3] = QMDDEdge(exp(i * c) * cos(a - b), nullptr);

    nNode->edges[0] = QMDDEdge(1.0, nNode1);
    nNode->edges[1] = QMDDEdge(i, nNode2);
    nNode->edges[2] = QMDDEdge(i, nNode3);
    nNode->edges[3] = QMDDEdge(1.0, nNode4);

    QMDDEdge nEdge(nWeight, nNode);
    return QMDDGate(nEdge);
}



// QMDDGate createSDaggerGate() {
//     complex<double> sDaggerWeight = 1.0;
//     auto sDaggerNode = make_shared<QMDDNode>(4);

//     sDaggerNode->edges[0] = QMDDEdge(1, nullptr);
//     sDaggerNode->edges[1] = QMDDEdge(0, nullptr);
//     sDaggerNode->edges[2] = QMDDEdge(0, nullptr);
//     sDaggerNode->edges[3] = QMDDEdge(-i, nullptr);

//     QMDDEdge sDaggerEdge(sDaggerWeight, sDaggerNode);
//     return QMDDGate(sDaggerEdge);
// }

// QMDDGate createTDaggerGate() {
//     complex<double> tDaggerWeight = 1.0;
//     QMDDNode* tDaggerNode = new QMDDNode(4);

//     tDaggerNode->edges[0] = QMDDEdge(1, nullptr);
//     tDaggerNode->edges[1] = QMDDEdge(0, nullptr);
//     tDaggerNode->edges[2] = QMDDEdge(0, nullptr);
//     tDaggerNode->edges[3] = QMDDEdge(-exp(i * complex<double>(M_PI / 4)), nullptr);

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