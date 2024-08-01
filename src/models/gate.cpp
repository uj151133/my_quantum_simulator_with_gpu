#include "gate.hpp"

#include <iostream>

namespace py = pybind11;

complex<double> i(0.0, 1.0);

QMDDGate gate::ZERO() {
    complex<double> zeroWeight = 0.0;
    auto zeroNode = make_shared<QMDDNode>(4);

    zeroNode->edges[0] = QMDDEdge(0.0, nullptr);
    zeroNode->edges[1] = QMDDEdge(0.0, nullptr);
    zeroNode->edges[2] = QMDDEdge(0.0, nullptr);
    zeroNode->edges[3] = QMDDEdge(0.0, nullptr);

    QMDDEdge zeroEdge(zeroWeight, zeroNode);
    return QMDDGate(zeroEdge);
};


QMDDGate gate::I() {
    complex<double> iWeight = 1.0;
    auto iNode = make_shared<QMDDNode>(4);

    iNode->edges[0] = QMDDEdge(1.0, nullptr);
    iNode->edges[1] = QMDDEdge(0.0, nullptr);
    iNode->edges[2] = QMDDEdge(0.0, nullptr);
    iNode->edges[3] = QMDDEdge(1.0, nullptr);

    QMDDEdge iEdge(iWeight, iNode);
    return QMDDGate(iEdge);
};

QMDDGate gate::Ph(double delta) {
    complex<double> phWeight = exp(i * delta);
    auto phNode = make_shared<QMDDNode>(4);

    phNode->edges[0] = QMDDEdge(1.0, nullptr);
    phNode->edges[1] = QMDDEdge(0.0, nullptr);
    phNode->edges[2] = QMDDEdge(0.0, nullptr);
    phNode->edges[3] = QMDDEdge(1.0, nullptr);

    QMDDEdge phEdge(phWeight, phNode);
    return QMDDGate(phEdge);
}

QMDDGate gate::X() {
    complex<double> xWeight = 1.0;
    auto xNode = make_shared<QMDDNode>(4);

    xNode->edges[0] = QMDDEdge(0.0, nullptr);
    xNode->edges[1] = QMDDEdge(1.0, nullptr);
    xNode->edges[2] = QMDDEdge(1.0, nullptr);
    xNode->edges[3] = QMDDEdge(0.0, nullptr);

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
    auto yNode = make_shared<QMDDNode>(4);

    yNode->edges[0] = QMDDEdge(0.0, nullptr);
    yNode->edges[1] = QMDDEdge(-1.0, nullptr);
    yNode->edges[2] = QMDDEdge(1.0, nullptr);
    yNode->edges[3] = QMDDEdge(0.0, nullptr);

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
    auto zNode = make_shared<QMDDNode>(4);

    zNode->edges[0] = QMDDEdge(1.0, nullptr);
    zNode->edges[1] = QMDDEdge(0.0, nullptr);
    zNode->edges[2] = QMDDEdge(0.0, nullptr);
    zNode->edges[3] = QMDDEdge(-1.0, nullptr);

    QMDDEdge zEdge(zWeight, zNode);
    return QMDDGate(zEdge);
};

QMDDGate gate::S() {
    complex<double> sWeight = 1.0;
    auto sNode = make_shared<QMDDNode>(4);

    sNode->edges[0] = QMDDEdge(1.0, nullptr);
    sNode->edges[1] = QMDDEdge(0.0, nullptr);
    sNode->edges[2] = QMDDEdge(0.0, nullptr);
    sNode->edges[3] = QMDDEdge(i, nullptr);

    QMDDEdge sEdge(sWeight, sNode);
    return QMDDGate(sEdge);
};

QMDDGate gate::V() {
    complex<double> vWeight = 1.0 / 2.0 + i / 2.0;
    auto vNode = make_shared<QMDDNode>(4);

    vNode->edges[0] = QMDDEdge(1.0, nullptr);
    vNode->edges[1] = QMDDEdge(i, nullptr);
    vNode->edges[2] = QMDDEdge(i, nullptr);
    vNode->edges[3] = QMDDEdge(1.0, nullptr);

    QMDDEdge vEdge(vWeight, vNode);
    return QMDDGate(vEdge);
};

QMDDGate gate::H() {
    complex<double> hWeight = 1.0 / sqrt(2.0);
    auto hNode = make_shared<QMDDNode>(4);

    hNode->edges[0] = QMDDEdge(1.0, nullptr);
    hNode->edges[1] = QMDDEdge(1.0, nullptr);
    hNode->edges[2] = QMDDEdge(1.0, nullptr);
    hNode->edges[3] = QMDDEdge(-1.0, nullptr);

    QMDDEdge hEdge(hWeight, hNode);
    return QMDDGate(hEdge);
};

QMDDGate gate::CX1() {
    complex<double> cx1Weight = 1.0;
    auto cx1Node = make_shared<QMDDNode>(4);

    cx1Node->edges[0] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()));
    cx1Node->edges[1] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    cx1Node->edges[2] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    cx1Node->edges[3] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::X().getStartNode()));

    QMDDEdge cx1Edge(cx1Weight, cx1Node);
    return QMDDGate(cx1Edge);
};

QMDDGate gate::CX2() {
    complex<double> cx2Weight = 1.0;
    auto cx2Node = make_shared<QMDDNode>(4);
    auto cx2Node1 = make_shared<QMDDNode>(4);
    auto cx2Node2 = make_shared<QMDDNode>(4);

    cx2Node1->edges[0] = QMDDEdge(1.0, nullptr);
    cx2Node1->edges[1] = QMDDEdge(0.0, nullptr);
    cx2Node1->edges[2] = QMDDEdge(0.0, nullptr);
    cx2Node1->edges[3] = QMDDEdge(0.0, nullptr);

    cx2Node2->edges[0] = QMDDEdge(0.0, nullptr);
    cx2Node2->edges[1] = QMDDEdge(0.0, nullptr);
    cx2Node2->edges[2] = QMDDEdge(0.0, nullptr);
    cx2Node2->edges[3] = QMDDEdge(1.0, nullptr);

    cx2Node->edges[0] = QMDDEdge(1.0, cx2Node1);
    cx2Node->edges[1] = QMDDEdge(1.0, cx2Node2);
    cx2Node->edges[2] = QMDDEdge(1.0, cx2Node2);
    cx2Node->edges[3] = QMDDEdge(1.0, cx2Node1);

    QMDDEdge cx2Edge(cx2Weight, cx2Node);
    return QMDDGate(cx2Edge);
}

QMDDGate gate::P(double phi) {
    complex<double> pWeight = 1.0;
    auto pNode = make_shared<QMDDNode>(4);

    pNode->edges[0] = QMDDEdge(1.0, nullptr);
    pNode->edges[1] = QMDDEdge(0.0, nullptr);
    pNode->edges[2] = QMDDEdge(0.0, nullptr);
    pNode->edges[3] = QMDDEdge(exp(i * phi), nullptr);

    QMDDEdge pEdge(pWeight, pNode);
    return QMDDGate(pEdge);
}

QMDDGate gate::varCX() {
    complex<double> varCXWeight = 1.0;
    auto varCXNode = make_shared<QMDDNode>(4);

    varCXNode->edges[0] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::X().getStartNode()));
    varCXNode->edges[1] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    varCXNode->edges[2] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    varCXNode->edges[3] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()));

    QMDDEdge varCXEdge(varCXWeight, varCXNode);
    return QMDDGate(varCXEdge);
}

QMDDGate gate::CZ() {
    complex<double> czWeight = 1.0;
    auto czNode = make_shared<QMDDNode>(4);

    czNode->edges[0] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()));
    czNode->edges[1] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    czNode->edges[2] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    czNode->edges[3] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::Z().getStartNode()));

    QMDDEdge czEdge(czWeight, czNode);
    return QMDDGate(czEdge);
}

QMDDGate gate::DCNOT() {
    complex<double> dcnotWeight = 1.0;
    auto dcnotNode = make_shared<QMDDNode>(4);
    auto dcnotNode1 = make_shared<QMDDNode>(4);
    auto dcnotNode2 = make_shared<QMDDNode>(4);
    auto dcnotNode3 = make_shared<QMDDNode>(4);
    auto dcnotNode4 = make_shared<QMDDNode>(4);

    dcnotNode1->edges[0] = QMDDEdge(1.0, nullptr);
    dcnotNode1->edges[1] = QMDDEdge(0.0, nullptr);
    dcnotNode1->edges[2] = QMDDEdge(0.0, nullptr);
    dcnotNode1->edges[3] = QMDDEdge(0.0, nullptr);

    dcnotNode2->edges[0] = QMDDEdge(0.0, nullptr);
    dcnotNode2->edges[1] = QMDDEdge(0.0, nullptr);
    dcnotNode2->edges[2] = QMDDEdge(1.0, nullptr);
    dcnotNode2->edges[3] = QMDDEdge(0.0, nullptr);

    dcnotNode3->edges[0] = QMDDEdge(0.0, nullptr);
    dcnotNode3->edges[1] = QMDDEdge(0.0, nullptr);
    dcnotNode3->edges[2] = QMDDEdge(0.0, nullptr);
    dcnotNode3->edges[3] = QMDDEdge(1.0, nullptr);

    dcnotNode4->edges[0] = QMDDEdge(0.0, nullptr);
    dcnotNode4->edges[1] = QMDDEdge(1.0, nullptr);
    dcnotNode4->edges[2] = QMDDEdge(0.0, nullptr);
    dcnotNode4->edges[3] = QMDDEdge(0.0, nullptr);

    dcnotNode->edges[0] = QMDDEdge(1.0, dcnotNode1);
    dcnotNode->edges[1] = QMDDEdge(1.0, dcnotNode2);
    dcnotNode->edges[2] = QMDDEdge(1.0, dcnotNode3);
    dcnotNode->edges[3] = QMDDEdge(1.0, dcnotNode4);

    QMDDEdge dcnotEdge(dcnotWeight, dcnotNode);
    return QMDDGate(dcnotEdge);
}

QMDDGate gate::SWAP() {
    complex<double> swapWeight = 1.0;
    auto swapNode = make_shared<QMDDNode>(4);
    auto swapNode1 = make_shared<QMDDNode>(4);
    auto swapNode2 = make_shared<QMDDNode>(4);
    auto swapNode3 = make_shared<QMDDNode>(4);
    auto swapNode4 = make_shared<QMDDNode>(4);

    swapNode1->edges[0] = QMDDEdge(1.0, nullptr);
    swapNode1->edges[1] = QMDDEdge(0.0, nullptr);
    swapNode1->edges[2] = QMDDEdge(0.0, nullptr);
    swapNode1->edges[3] = QMDDEdge(0.0, nullptr);

    swapNode2->edges[0] = QMDDEdge(0.0, nullptr);
    swapNode2->edges[1] = QMDDEdge(0.0, nullptr);
    swapNode2->edges[2] = QMDDEdge(1.0, nullptr);
    swapNode2->edges[3] = QMDDEdge(0.0, nullptr);

    swapNode3->edges[0] = QMDDEdge(0.0, nullptr);
    swapNode3->edges[1] = QMDDEdge(1.0, nullptr);
    swapNode3->edges[2] = QMDDEdge(0.0, nullptr);
    swapNode3->edges[3] = QMDDEdge(0.0, nullptr);

    swapNode4->edges[0] = QMDDEdge(0.0, nullptr);
    swapNode4->edges[1] = QMDDEdge(0.0, nullptr);
    swapNode4->edges[2] = QMDDEdge(0.0, nullptr);
    swapNode4->edges[3] = QMDDEdge(1.0, nullptr);

    swapNode->edges[0] = QMDDEdge(1.0, swapNode1);
    swapNode->edges[1] = QMDDEdge(1.0, swapNode2);
    swapNode->edges[2] = QMDDEdge(1.0, swapNode3);
    swapNode->edges[3] = QMDDEdge(1.0, swapNode4);

    QMDDEdge swapEdge(swapWeight, swapNode);
    return QMDDGate(swapEdge);
}

QMDDGate gate::iSWAP() {
    complex<double> iswapWeight = 1.0;
    auto iswapNode = make_shared<QMDDNode>(4);
    auto iswapNode1 = make_shared<QMDDNode>(4);
    auto iswapNode2 = make_shared<QMDDNode>(4);
    auto iswapNode3 = make_shared<QMDDNode>(4);
    auto iswapNode4 = make_shared<QMDDNode>(4);

    iswapNode1->edges[0] = QMDDEdge(1.0, nullptr);
    iswapNode1->edges[1] = QMDDEdge(0.0, nullptr);
    iswapNode1->edges[2] = QMDDEdge(0.0, nullptr);
    iswapNode1->edges[3] = QMDDEdge(0.0, nullptr);

    iswapNode2->edges[0] = QMDDEdge(0.0, nullptr);
    iswapNode2->edges[1] = QMDDEdge(0.0, nullptr);
    iswapNode2->edges[2] = QMDDEdge(1.0, nullptr);
    iswapNode2->edges[3] = QMDDEdge(0.0, nullptr);

    iswapNode3->edges[0] = QMDDEdge(0.0, nullptr);
    iswapNode3->edges[1] = QMDDEdge(1.0, nullptr);
    iswapNode3->edges[2] = QMDDEdge(0.0, nullptr);
    iswapNode3->edges[3] = QMDDEdge(0.0, nullptr);

    iswapNode4->edges[0] = QMDDEdge(0.0, nullptr);
    iswapNode4->edges[1] = QMDDEdge(0.0, nullptr);
    iswapNode4->edges[2] = QMDDEdge(0.0, nullptr);
    iswapNode4->edges[3] = QMDDEdge(1.0, nullptr);

    iswapNode->edges[0] = QMDDEdge(1.0, iswapNode1);
    iswapNode->edges[1] = QMDDEdge(i, iswapNode2);
    iswapNode->edges[2] = QMDDEdge(i, iswapNode3);
    iswapNode->edges[3] = QMDDEdge(1.0, iswapNode4);

    QMDDEdge iswapEdge(iswapWeight, iswapNode);
    return QMDDGate(iswapEdge);
}

QMDDGate gate::T() {
    complex<double> tWeight = 1.0;
    auto tNode = make_shared<QMDDNode>(4);

    tNode->edges[0] = QMDDEdge(1.0, nullptr);
    tNode->edges[1] = QMDDEdge(0.0, nullptr);
    tNode->edges[2] = QMDDEdge(0.0, nullptr);
    tNode->edges[3] = QMDDEdge(exp(i * complex<double>(M_PI / 4.0)), nullptr);

    QMDDEdge tEdge(tWeight, tNode);
    return QMDDGate(tEdge);
};

QMDDGate gate::CP(double phi) {
    complex<double> cpWeight = 1.0;
    auto cpNode = make_shared<QMDDNode>(4);

    cpNode->edges[0] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()));
    cpNode->edges[1] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    cpNode->edges[2] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    cpNode->edges[3] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::P(phi).getStartNode()));

    QMDDEdge cpEdge(cpWeight, cpNode);
    return QMDDGate(cpEdge);
}

QMDDGate gate::CS() {
    complex<double> csWeight = 1.0;
    auto csNode = make_shared<QMDDNode>(4);

    csNode->edges[0] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()));
    csNode->edges[1] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    csNode->edges[2] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    csNode->edges[3] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::S().getStartNode()));

    QMDDEdge csEdge(csWeight, csNode);
    return QMDDGate(csEdge);
}

QMDDGate gate::Rx(double theta) {
    complex<double> rxWeight = 1.0;
    auto rxNode = make_shared<QMDDNode>(4);

    rxNode->edges[0] = QMDDEdge(cos(theta / 2.0), nullptr);
    rxNode->edges[1] = QMDDEdge(-i * sin(theta / 2.0), nullptr);
    rxNode->edges[2] = QMDDEdge(-i * sin(theta / 2.0), nullptr);
    rxNode->edges[3] = QMDDEdge(cos(theta / 2.0), nullptr);

    QMDDEdge rxEdge(rxWeight, rxNode);
    return QMDDGate(rxEdge);
}

QMDDGate gate::Ry(double theta) {
    complex<double> ryWeight = 1.0;
    auto ryNode = make_shared<QMDDNode>(4);

    ryNode->edges[0] = QMDDEdge(cos(theta / 2.0), nullptr);
    ryNode->edges[1] = QMDDEdge(-sin(theta / 2.0), nullptr);
    ryNode->edges[2] = QMDDEdge(sin(theta / 2.0), nullptr);
    ryNode->edges[3] = QMDDEdge(cos(theta / 2.0), nullptr);

    QMDDEdge ryEdge(ryWeight, ryNode);
    return QMDDGate(ryEdge);
}

QMDDGate gate::Rz(double theta) {
    complex<double> rzWeight = 1.0;
    auto rzNode = make_shared<QMDDNode>(4);

    rzNode->edges[0] = QMDDEdge(exp(-i * theta / 2.0), nullptr);
    rzNode->edges[1] = QMDDEdge(0.0, nullptr);
    rzNode->edges[2] = QMDDEdge(0.0, nullptr);
    rzNode->edges[3] = QMDDEdge(exp(i * theta / 2.0), nullptr);

    QMDDEdge rzEdge(rzWeight, rzNode);
    return QMDDGate(rzEdge);
}

QMDDGate gate::Rxx(double phi) {
    complex<double> rxxWeight = 1.0;
    auto rxxNode = make_shared<QMDDNode>(4);

    rxxNode->edges[0] = QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode()));
    rxxNode->edges[1] = QMDDEdge(-i * sin(phi / 2.0), shared_ptr<QMDDNode>(gate::X().getStartNode()));
    rxxNode->edges[2] = QMDDEdge(-i * sin(phi / 2.0), shared_ptr<QMDDNode>(gate::X().getStartNode()));
    rxxNode->edges[3] = QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode()));

    QMDDEdge rxxEdge(rxxWeight, rxxNode);
    return QMDDGate(rxxEdge);
}

QMDDGate gate::Ryy(double phi) {
    complex<double> ryyWeight = 1.0;
    auto ryyNode = make_shared<QMDDNode>(4);

    ryyNode->edges[0] = QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode()));
    ryyNode->edges[1] = QMDDEdge(-sin(phi / 2.0), shared_ptr<QMDDNode>(gate::Y().getStartNode()));
    ryyNode->edges[2] = QMDDEdge(sin(phi / 2.0), shared_ptr<QMDDNode>(gate::Y().getStartNode()));
    ryyNode->edges[3] = QMDDEdge(cos(phi / 2.0), shared_ptr<QMDDNode>(gate::I().getStartNode()));

    QMDDEdge ryyEdge(ryyWeight, ryyNode);
    return QMDDGate(ryyEdge);
}

QMDDGate gate::Rzz(double phi) {
    complex<double> rzzWeight = 1.0;
    auto rzzNode = make_shared<QMDDNode>(4);

    rzzNode->edges[0] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::Rz(phi).getStartNode()));
    rzzNode->edges[1] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    rzzNode->edges[2] = QMDDEdge(0.0, shared_ptr<QMDDNode>(gate::ZERO().getStartNode()));
    rzzNode->edges[3] = QMDDEdge(exp(i * phi / 2.0), shared_ptr<QMDDNode>(gate::P(-phi).getStartNode()));

    QMDDEdge rzzEdge(rzzWeight, rzzNode);
    return QMDDGate(rzzEdge);
}

QMDDGate gate::Rxy(double phi) {
    complex<double> rxyWeight = 1.0;
    auto rxyNode = make_shared<QMDDNode>(4);
    auto rxyNode1 = make_shared<QMDDNode>(4);
    auto rxyNode2 = make_shared<QMDDNode>(4);
    auto rxyNode3 = make_shared<QMDDNode>(4);
    auto rxyNode4 = make_shared<QMDDNode>(4);

    rxyNode1->edges[0] = QMDDEdge(1.0, nullptr);
    rxyNode1->edges[1] = QMDDEdge(0.0, nullptr);
    rxyNode1->edges[2] = QMDDEdge(0.0, nullptr);
    rxyNode1->edges[3] = QMDDEdge(cos(phi / 2.0), nullptr);

    rxyNode2->edges[0] = QMDDEdge(0.0, nullptr);
    rxyNode2->edges[1] = QMDDEdge(0.0, nullptr);
    rxyNode2->edges[2] = QMDDEdge(-i * sin(phi / 2.0), nullptr);
    rxyNode2->edges[3] = QMDDEdge(0.0, nullptr);

    rxyNode3->edges[0] = QMDDEdge(0.0, nullptr);
    rxyNode3->edges[1] = QMDDEdge(-i * sin(phi / 2.0), nullptr);
    rxyNode3->edges[2] = QMDDEdge(0.0, nullptr);
    rxyNode3->edges[3] = QMDDEdge(0.0, nullptr);

    rxyNode4->edges[0] = QMDDEdge(cos(phi / 2.0), nullptr);
    rxyNode4->edges[1] = QMDDEdge(0.0, nullptr);
    rxyNode4->edges[2] = QMDDEdge(0.0, nullptr);
    rxyNode4->edges[3] = QMDDEdge(1.0, nullptr);

    rxyNode->edges[0] = QMDDEdge(1.0, rxyNode1);
    rxyNode->edges[1] = QMDDEdge(1.0, rxyNode2);
    rxyNode->edges[2] = QMDDEdge(1.0, rxyNode3);
    rxyNode->edges[3] = QMDDEdge(1.0, rxyNode4);

    QMDDEdge rxyEdge(rxyWeight, rxyNode);
    return QMDDGate(rxyEdge);
}

QMDDGate gate::SquareSWAP() {
    complex<double> squareSWAPWeight = 1.0;
    auto squareSWAPNode = make_shared<QMDDNode>(4);
    auto squareSWAPNode1 = make_shared<QMDDNode>(4);
    auto squareSWAPNode2 = make_shared<QMDDNode>(4);
    auto squareSWAPNode3 = make_shared<QMDDNode>(4);
    auto squareSWAPNode4 = make_shared<QMDDNode>(4);

    squareSWAPNode1->edges[0] = QMDDEdge(1.0, nullptr);
    squareSWAPNode1->edges[1] = QMDDEdge(0.0, nullptr);
    squareSWAPNode1->edges[2] = QMDDEdge(0.0, nullptr);
    squareSWAPNode1->edges[3] = QMDDEdge((1.0 + i) / 2.0, nullptr);

    squareSWAPNode2->edges[0] = QMDDEdge(0.0, nullptr);
    squareSWAPNode2->edges[1] = QMDDEdge(0.0, nullptr);
    squareSWAPNode2->edges[2] = QMDDEdge(1.0, nullptr);
    squareSWAPNode2->edges[3] = QMDDEdge(0.0, nullptr);

    squareSWAPNode3->edges[0] = QMDDEdge(0.0, nullptr);
    squareSWAPNode3->edges[1] = QMDDEdge(1.0, nullptr);
    squareSWAPNode3->edges[2] = QMDDEdge(0.0, nullptr);
    squareSWAPNode3->edges[3] = QMDDEdge(0.0, nullptr);

    squareSWAPNode4->edges[0] = QMDDEdge((1.0 + i) / 2.0, nullptr);
    squareSWAPNode4->edges[1] = QMDDEdge(0.0, nullptr);
    squareSWAPNode4->edges[2] = QMDDEdge(0.0, nullptr);
    squareSWAPNode4->edges[3] = QMDDEdge(1.0, nullptr);

    squareSWAPNode->edges[0] = QMDDEdge(1.0, squareSWAPNode1);
    squareSWAPNode->edges[1] = QMDDEdge((1.0 - i) / 2.0, squareSWAPNode2);
    squareSWAPNode->edges[2] = QMDDEdge((1.0 - i) / 2.0, squareSWAPNode3);
    squareSWAPNode->edges[3] = QMDDEdge(1.0, squareSWAPNode4);

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