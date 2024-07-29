#include "gate.hpp"



#include <iostream>

namespace py = pybind11;

complex<double> i(0.0, 1.0);

QMDDGate gate::I() {
    complex<double> iWeight = 1.0;
    auto iNode = make_shared<QMDDNode>(4);

    iNode->edges[0] = QMDDEdge(1, nullptr);
    iNode->edges[1] = QMDDEdge(0, nullptr);
    iNode->edges[2] = QMDDEdge(0, nullptr);
    iNode->edges[3] = QMDDEdge(1, nullptr);

    QMDDEdge iEdge(iWeight, iNode);
    return QMDDGate(iEdge);
};

QMDDGate gate::Ph(double delta) {
    complex<double> phWeight = exp(i * delta);
    auto phNode = make_shared<QMDDNode>(4);

    phNode->edges[0] = QMDDEdge(1, nullptr);
    phNode->edges[1] = QMDDEdge(0, nullptr);
    phNode->edges[2] = QMDDEdge(0, nullptr);
    phNode->edges[3] = QMDDEdge(1, nullptr);

    QMDDEdge phEdge(phWeight, phNode);
    return QMDDGate(phEdge);
}

QMDDGate gate::X() {
    complex<double> xWeight = 1.0;
    auto xNode = make_shared<QMDDNode>(4);

    xNode->edges[0] = QMDDEdge(0, nullptr);
    xNode->edges[1] = QMDDEdge(1, nullptr);
    xNode->edges[2] = QMDDEdge(1, nullptr);
    xNode->edges[3] = QMDDEdge(0, nullptr);

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

    yNode->edges[0] = QMDDEdge(0, nullptr);
    yNode->edges[1] = QMDDEdge(-1, nullptr);
    yNode->edges[2] = QMDDEdge(1, nullptr);
    yNode->edges[3] = QMDDEdge(0, nullptr);

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

    zNode->edges[0] = QMDDEdge(1, nullptr);
    zNode->edges[1] = QMDDEdge(0, nullptr);
    zNode->edges[2] = QMDDEdge(0, nullptr);
    zNode->edges[3] = QMDDEdge(-1, nullptr);

    QMDDEdge zEdge(zWeight, zNode);
    return QMDDGate(zEdge);
};

QMDDGate gate::S() {
    complex<double> sWeight = 1.0;
    auto sNode = make_shared<QMDDNode>(4);

    sNode->edges[0] = QMDDEdge(1, nullptr);
    sNode->edges[1] = QMDDEdge(0, nullptr);
    sNode->edges[2] = QMDDEdge(0, nullptr);
    sNode->edges[3] = QMDDEdge(i, nullptr);

    QMDDEdge sEdge(sWeight, sNode);
    return QMDDGate(sEdge);
};

QMDDGate gate::V() {
    complex<double> vWeight = 1.0 / 2.0 + i / 2.0;
    auto vNode = make_shared<QMDDNode>(4);

    vNode->edges[0] = QMDDEdge(1, nullptr);
    vNode->edges[1] = QMDDEdge(i, nullptr);
    vNode->edges[2] = QMDDEdge(i, nullptr);
    vNode->edges[3] = QMDDEdge(1, nullptr);

    QMDDEdge vEdge(vWeight, vNode);
    return QMDDGate(vEdge);
};



QMDDGate gate::H() {
    complex<double> hWeight = 1.0 / sqrt(2.0);
    auto hNode = make_shared<QMDDNode>(4);

    hNode->edges[0] = QMDDEdge(1, nullptr);
    hNode->edges[1] = QMDDEdge(1, nullptr);
    hNode->edges[2] = QMDDEdge(1, nullptr);
    hNode->edges[3] = QMDDEdge(-1, nullptr);

    QMDDEdge hEdge(hWeight, hNode);
    return QMDDGate(hEdge);
};

QMDDGate gate::P(double phi) {
    complex<double> pWeight = 1.0;
    auto pNode = make_shared<QMDDNode>(4);

    pNode->edges[0] = QMDDEdge(1, nullptr);
    pNode->edges[1] = QMDDEdge(0, nullptr);
    pNode->edges[2] = QMDDEdge(0, nullptr);
    pNode->edges[3] = QMDDEdge(exp(i * phi), nullptr);

    QMDDEdge pEdge(pWeight, pNode);
    return QMDDGate(pEdge);
}


QMDDGate gate::T() {
    complex<double> tWeight = 1.0;
    auto tNode = make_shared<QMDDNode>(4);

    tNode->edges[0] = QMDDEdge(1, nullptr);
    tNode->edges[1] = QMDDEdge(0, nullptr);
    tNode->edges[2] = QMDDEdge(0, nullptr);
    tNode->edges[3] = QMDDEdge(exp(i * complex<double>(M_PI / 4)), nullptr);

    QMDDEdge tEdge(tWeight, tNode);
    return QMDDGate(tEdge);
};

QMDDGate gate::Rx(double theta) {
    complex<double> rxWeight = 1.0;
    auto rxNode = make_shared<QMDDNode>(4);

    rxNode->edges[0] = QMDDEdge(cos(theta / 2), nullptr);
    rxNode->edges[1] = QMDDEdge(-i * sin(theta / 2), nullptr);
    rxNode->edges[2] = QMDDEdge(-i * sin(theta / 2), nullptr);
    rxNode->edges[3] = QMDDEdge(cos(theta / 2), nullptr);

    QMDDEdge rxEdge(rxWeight, rxNode);
    return QMDDGate(rxEdge);
}

QMDDGate gate::Ry(double theta) {
    complex<double> ryWeight = 1.0;
    auto ryNode = make_shared<QMDDNode>(4);

    ryNode->edges[0] = QMDDEdge(cos(theta / 2), nullptr);
    ryNode->edges[1] = QMDDEdge(-sin(theta / 2), nullptr);
    ryNode->edges[2] = QMDDEdge(sin(theta / 2), nullptr);
    ryNode->edges[3] = QMDDEdge(cos(theta / 2), nullptr);

    QMDDEdge ryEdge(ryWeight, ryNode);
    return QMDDGate(ryEdge);
}

QMDDGate gate::Rz(double theta) {
    complex<double> rzWeight = 1.0;
    auto rzNode = make_shared<QMDDNode>(4);

    rzNode->edges[0] = QMDDEdge(exp(-i * theta / 2.0), nullptr);
    rzNode->edges[1] = QMDDEdge(0, nullptr);
    rzNode->edges[2] = QMDDEdge(0, nullptr);
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
    rzzNode->edges[1] = QMDDEdge(0, nullptr);
    rzzNode->edges[2] = QMDDEdge(0, nullptr);
    rzzNode->edges[3] = QMDDEdge(exp(i * phi / 2.0), shared_ptr<QMDDNode>(gate::P(-phi).getStartNode()));

    QMDDEdge rzzEdge(rzzWeight, rzzNode);
    return QMDDGate(rzzEdge);
}

QMDDGate gate::Rxy(double phi) {
    complex<double> rxyWeight = 1.0;
    auto rxyNode = make_shared<QMDDNode>(4);
    auto rxy1Node = make_shared<QMDDNode>(4);
    auto rxy2Node = make_shared<QMDDNode>(4);
    auto rxy3Node = make_shared<QMDDNode>(4);
    auto rxy4Node = make_shared<QMDDNode>(4);

    rxy1Node->edges[0] = QMDDEdge(1.0, nullptr);
    rxy1Node->edges[1] = QMDDEdge(0, nullptr);
    rxy1Node->edges[2] = QMDDEdge(0, nullptr);
    rxy1Node->edges[3] = QMDDEdge(cos(phi / 2.0), nullptr);

    rxy2Node->edges[0] = QMDDEdge(0, nullptr);
    rxy2Node->edges[1] = QMDDEdge(0, nullptr);
    rxy2Node->edges[2] = QMDDEdge(-i * sin(phi / 2.0), nullptr);
    rxy2Node->edges[3] = QMDDEdge(0, nullptr);

    rxy3Node->edges[0] = QMDDEdge(0, nullptr);
    rxy3Node->edges[1] = QMDDEdge(-i * sin(phi / 2.0), nullptr);
    rxy3Node->edges[2] = QMDDEdge(0, nullptr);
    rxy3Node->edges[3] = QMDDEdge(0, nullptr);

    rxy4Node->edges[0] = QMDDEdge(cos(phi / 2.0), nullptr);
    rxy4Node->edges[1] = QMDDEdge(0, nullptr);
    rxy4Node->edges[2] = QMDDEdge(0, nullptr);
    rxy4Node->edges[3] = QMDDEdge(1.0, nullptr);

    rxyNode->edges[0] = QMDDEdge(1.0, rxy1Node);
    rxyNode->edges[1] = QMDDEdge(1.0, rxy2Node);
    rxyNode->edges[2] = QMDDEdge(1.0, rxy3Node);
    rxyNode->edges[3] = QMDDEdge(1.0, rxy4Node);

    QMDDEdge rxyEdge(rxyWeight, rxyNode);
    return QMDDGate(rxyEdge);
}

QMDDGate gate::U(double theta, double phi, double lambda) {
    complex<double> uWeight = 1.0;
    auto uNode = make_shared<QMDDNode>(4);

    uNode->edges[0] = QMDDEdge(cos(theta / 2), nullptr);
    uNode->edges[1] = QMDDEdge(-exp(i * lambda) * sin(theta / 2), nullptr);
    uNode->edges[2] = QMDDEdge(exp(i * phi) * sin(theta / 2), nullptr);
    uNode->edges[3] = QMDDEdge(exp(i * (phi + lambda)) * cos(theta / 2), nullptr);

    QMDDEdge uEdge(uWeight, uNode);
    return QMDDGate(uEdge);
}

QMDDGate gate::BARENCO(double alpha, double phi, double theta) {
    complex<double> barencoWeight = 1.0;
    auto barencoNode = make_shared<QMDDNode>(4);
    auto barenco1Node = make_shared<QMDDNode>(4);

    barenco1Node->edges[0] = QMDDEdge(exp(i * alpha) * cos(theta), nullptr);
    barenco1Node->edges[1] = QMDDEdge(-i * exp(i * (alpha - phi)) * sin(theta), nullptr);
    barenco1Node->edges[2] = QMDDEdge(-i * exp(i * (alpha + phi)) * sin(theta), nullptr);
    barenco1Node->edges[3] = QMDDEdge(exp(i * alpha) * cos(theta), nullptr);

    barencoNode->edges[0] = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate::I().getStartNode()));
    barencoNode->edges[1] = QMDDEdge(0.0, nullptr);
    barencoNode->edges[2] = QMDDEdge(0.0, nullptr);
    barencoNode->edges[3] = QMDDEdge(1.0, barenco1Node);

    QMDDEdge barencoEdge(barencoWeight, barencoNode);
    return QMDDGate(barencoEdge);
}

QMDDGate gate::B() {
    complex<double> bWeight = 1.0;
    auto bNode = make_shared<QMDDNode>(4);
    auto b1Node = make_shared<QMDDNode>(4);
    auto b2Node = make_shared<QMDDNode>(4);

    b1Node->edges[0] = QMDDEdge(cos(M_PI / 8.0), nullptr);
    b1Node->edges[1] = QMDDEdge(0, nullptr);
    b1Node->edges[2] = QMDDEdge(0, nullptr);
    b1Node->edges[3] = QMDDEdge(cos(3.0 * M_PI / 8.0), nullptr);

    b2Node->edges[0] = QMDDEdge(0, nullptr);
    b2Node->edges[1] = QMDDEdge(sin(M_PI / 8.0), nullptr);
    b2Node->edges[2] = QMDDEdge(sin(3.0 * M_PI / 8.0), nullptr);
    b2Node->edges[3] = QMDDEdge(0, nullptr);

    bNode->edges[0] = QMDDEdge(1.0, b1Node);
    bNode->edges[1] = QMDDEdge(i, b2Node);
    bNode->edges[2] = QMDDEdge(i * sin(M_PI / 8.0), shared_ptr<QMDDNode>(gate::X().getStartNode()));
    bNode->edges[3] = QMDDEdge(cos(M_PI / 8.0), shared_ptr<QMDDNode>(gate::I().getStartNode()));

    QMDDEdge bEdge(bWeight, bNode);
    return QMDDGate(bEdge);
}

QMDDGate gate::N(double a, double b, double c) {
    complex<double> nWeight = 1.0;
    auto nNode = make_shared<QMDDNode>(4);
    auto n1Node = make_shared<QMDDNode>(4);
    auto n2Node = make_shared<QMDDNode>(4);
    auto n3Node = make_shared<QMDDNode>(4);
    auto n4Node = make_shared<QMDDNode>(4);

    n1Node->edges[0] = QMDDEdge(exp(i * c) * cos(a - b), nullptr);
    n1Node->edges[1] = QMDDEdge(0, nullptr);
    n1Node->edges[2] = QMDDEdge(0, nullptr);
    n1Node->edges[3] = QMDDEdge(exp(-i * c) * cos(a + b), nullptr);

    n2Node->edges[0] = QMDDEdge(0, nullptr);
    n2Node->edges[1] = QMDDEdge(exp(i * c) * sin(a - b), nullptr);
    n2Node->edges[2] = QMDDEdge(exp(-i * c) * sin(a + b), nullptr);
    n2Node->edges[3] = QMDDEdge(0, nullptr);

    n3Node->edges[0] = QMDDEdge(0, nullptr);
    n3Node->edges[1] = QMDDEdge(exp(-i * c) * sin(a + b), nullptr);
    n3Node->edges[2] = QMDDEdge(exp(i * c) * sin(a - b), nullptr);
    n3Node->edges[3] = QMDDEdge(0, nullptr);

    n4Node->edges[0] = QMDDEdge(exp(-i * c) * cos(a + b), nullptr);
    n4Node->edges[1] = QMDDEdge(0, nullptr);
    n4Node->edges[2] = QMDDEdge(0, nullptr);
    n4Node->edges[3] = QMDDEdge(exp(i * c) * cos(a - b), nullptr);

    nNode->edges[0] = QMDDEdge(1.0, n1Node);
    nNode->edges[1] = QMDDEdge(i, n2Node);
    nNode->edges[2] = QMDDEdge(i, n3Node);
    nNode->edges[3] = QMDDEdge(1.0, n4Node);

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

// const matrix CNOT_GATE = matrix{
//     {1, 0, 0, 0}, 
//     {0, 1, 0, 0},
//     {0, 0, 0, 1},
//     {0, 0, 1, 0}
// };

// const matrix CZ_GATE = matrix{ 
//     {1, 0, 0, 0},
//     {0, 1, 0, 0},
//     {0, 0, 1, 0},
//     {0, 0, 0, -1}
// };

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

// const matrix SWAP_GATE = matrix{ 
//     {1, 0, 0, 0},
//     {0, 0, 1, 0},
//     {0, 1, 0, 0}, 
//     {0, 0, 0, 1}
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