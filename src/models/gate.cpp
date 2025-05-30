#include "gate.hpp"

ostream& operator<<(ostream& os, Type type) {
    switch (type) {
        case Type::I: os << "I"; break;
        case Type::Ph: os << "Ph"; break;
        case Type::X: os << "X"; break;
        case Type::Y: os << "Y"; break;
        case Type::Z: os << "Z"; break;
        case Type::S: os << "S"; break;
        case Type::Sdg: os << "Sdg"; break;
        case Type::V: os << "V"; break;
        case Type::Vdg: os << "Vdg"; break;
        case Type::H: os << "H"; break;
        case Type::CX: os << "CX"; break;
        case Type::varCX: os << "varCX"; break;
        case Type::CZ: os << "CZ"; break;
        case Type::SWAP: os << "SWAP"; break;
        case Type::P: os << "P"; break;
        case Type::T: os << "T"; break;
        case Type::Tdg: os << "Tdg"; break;
        case Type::CP: os << "CP"; break;
        case Type::CS: os << "CS"; break;
        case Type::R: os << "R"; break;
        case Type::Rx: os << "Rx"; break;
        case Type::Ry: os << "Ry"; break;
        case Type::Rz: os << "Rz"; break;
        case Type::Rxx: os << "Rxx"; break;
        case Type::Ryy: os << "Ryy"; break;
        case Type::Rzz: os << "Rzz"; break;
        case Type::Rxy: os << "Rxy"; break;
        case Type::U: os << "U"; break;
        case Type::U1: os << "U1"; break;
        case Type::U2: os << "U2"; break;
        case Type::U3: os << "U3"; break;
        case Type::Other: os << "Other"; break;
        case Type::Void: os << "Void"; break;
        default: os << "Unknown"; break;
    }
    return os;
}


QMDDGate gate::I() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeOne}
    })));
}

QMDDGate gate::Ph(double delta) {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(exp(i * delta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeOne}
    })));
}

QMDDGate gate::X() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeOne, edgeZero}
    })));
}

// const QMDDGate gate::PLUS_X_GATE = [] {
//     complex<double> plusXWeight = 1 / M_SQRT2;
//     QMDDEdge plusXNode = make_shared<QMDDNode>(4);

//     plusXNode->children[0] = QMDDEdge(1, nullptr);
//     plusXNode->children[1] = QMDDEdge(i, nullptr);
//     plusXNode->children[2] = QMDDEdge(i, nullptr);
//     plusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusXEdge(plusXWeight, plusXNode);
//     return QMDDGate(plusXEdge);
// }();

// const QMDDGate gate::MINUS_X_GATE = [] {
//     complex<double> minusXWeight = 1 / M_SQRT2;
//     QMDDEdge minusXNode = make_shared<QMDDNode>(4);

//     minusXNode->children[0] = QMDDEdge(1, nullptr);
//     minusXNode->children[1] = QMDDEdge(-i, nullptr);
//     minusXNode->children[2] = QMDDEdge(-i, nullptr);
//     minusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusXEdge(minusXWeight, minusXNode);
//     return QMDDGate(minusXEdge);
// }();

QMDDGate gate::Y() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(-i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(-1.0, nullptr), edgeZero}
    })));
}

// QMDDGate createPlusYGate() {
//     complex<double> plusYWeight = 1 / M_SQRT2;
//     QMDDNode* plusYNode = new QMDDNode(4);

//     plusYNode->children[0] = QMDDEdge(1, nullptr);
//     plusYNode->children[1] = QMDDEdge(1, nullptr);
//     plusYNode->children[2] = QMDDEdge(-1, nullptr);
//     plusYNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusYEdge(plusYWeight, plusYNode);
//     return QMDDGate(plusYEdge);
// }

// QMDDGate createMinusYGate() {
//     complex<double> minusYWeight = 1 / M_SQRT2;
//     QMDDNode* minusYNode = new QMDDNode(4);

//     minusYNode->children[0] = QMDDEdge(1, nullptr);
//     minusYNode->children[1] = QMDDEdge(-1, nullptr);
//     minusYNode->children[2] = QMDDEdge(1, nullptr);
//     minusYNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusYEdge(minusYWeight, minusYNode);
//     return QMDDGate(minusYEdge);
// }
QMDDGate gate::Z() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(-1.0, nullptr)}
    })));
}

QMDDGate gate::S() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(i, nullptr)}
    })));
}

QMDDGate gate::Sdg() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(-i, nullptr)}
    })));
}

QMDDGate gate::V() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge vEdge = QMDDEdge(i, nullptr);

    return QMDDGate(QMDDEdge(1.0 / 2.0 + i / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, vEdge},
        {vEdge, edgeOne}
    })));
}

QMDDGate gate::Vdg() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge vdgEdge = QMDDEdge(i, nullptr);

    return QMDDGate(QMDDEdge(1.0 / 2.0 - i / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, vdgEdge},
        {vdgEdge, edgeOne}
    })));
}

QMDDGate gate::H() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeOne},
        {edgeOne, QMDDEdge(-1.0, nullptr)}
    })));
}

QMDDGate gate::CX1() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge() , edgeZero},
        {edgeZero, gate::X().getInitialEdge()}
    })));
}

QMDDGate gate::CX2() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge cx2Edge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge cx2Edge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {cx2Edge1, cx2Edge2},
        {cx2Edge2, cx2Edge1}
    })));
}

QMDDGate gate::varCX() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::X().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::CZ() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::Z().getInitialEdge()}
    })));
}

QMDDGate gate::DCNOT() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge dcnotEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge dcnotEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge dcnotEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    QMDDEdge dcnotEdge4 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dcnotEdge1, dcnotEdge2},
        {dcnotEdge3, dcnotEdge4}
    })));
}

QMDDGate gate::SWAP() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge swapEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge swapEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge swapEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge swapEdge4 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {swapEdge1, swapEdge2},
        {swapEdge3, swapEdge4}
    })));
}

QMDDGate gate::iSWAP() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge iswapEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge iswapEdge2 = QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge iswapEdge3 = QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge iswapEdge4 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {iswapEdge1, iswapEdge2},
        {iswapEdge3, iswapEdge4}
    })));
}

QMDDGate gate::P(double phi) {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * phi), nullptr)}
    })));
}

QMDDGate gate::T() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * M_PI_4), nullptr)}
    })));
}

QMDDGate gate::Tdg() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(-i * M_PI_4), nullptr)}
    })));
}

QMDDGate gate::CP(double phi) {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::P(phi).getInitialEdge()}
    })));
}

QMDDGate gate::CS() {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::S().getInitialEdge()}
    })));
}

QMDDGate gate::R(double theta, double phi) {
    call_once(initEdgeFlag, initEdge);
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i * exp(- i * phi) * tanThetaHalf, nullptr)},
        {QMDDEdge(-i * exp(i * phi) * tanThetaHalf, nullptr), edgeOne}
    })));
}

QMDDGate gate::Rx(double theta) {
    call_once(initEdgeFlag, initEdge);
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i * tanThetaHalf, nullptr)},
        {QMDDEdge(-i * tanThetaHalf, nullptr), edgeOne}
    })));
}

QMDDGate gate::Ry(double theta) {
    call_once(initEdgeFlag, initEdge);
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-tanThetaHalf, nullptr)},
        {QMDDEdge(tanThetaHalf, nullptr), edgeOne}
    })));
}

QMDDGate gate::Rz(double theta) {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(exp(-i * theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * theta), nullptr)}
    })));
}

QMDDGate gate::Rk(int k) {
    call_once(initEdgeFlag, initEdge);
    complex<double> theta = 2 * M_PI * i / pow(2, k);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(theta), nullptr)}
    })));
}

QMDDGate gate::Rxx(double phi) {
    call_once(initEdgeFlag, initEdge);
    double phiHalf = phi / 2.0;
    double tanPhiHalf = tan(phiHalf);

    return QMDDGate(QMDDEdge(cos(phiHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), QMDDEdge(-i * tanPhiHalf, make_shared<QMDDNode>(*gate::X().getStartNode()))},
        {QMDDEdge(-i * tanPhiHalf, make_shared<QMDDNode>(*gate::X().getStartNode())), gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::Ryy(double phi) {
    call_once(initEdgeFlag, initEdge);
    double phiHalf = phi / 2.0;
    double tanPhiHalf = tan(phiHalf);

    return QMDDGate(QMDDEdge(cos(phiHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), QMDDEdge(i * tanPhiHalf, make_shared<QMDDNode>(*gate::Y().getStartNode()))},
        {QMDDEdge(-i * tanPhiHalf, make_shared<QMDDNode>(*gate::Y().getStartNode())), gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::Rzz(double phi) {
    call_once(initEdgeFlag, initEdge);
    return QMDDGate(QMDDEdge(exp(-i * phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::P(phi).getInitialEdge(), edgeZero},
        {edgeZero, QMDDEdge(exp(i * phi), make_shared<QMDDNode>(*gate::P(-phi).getStartNode()))}
    })));
}

QMDDGate gate::Rxy(double phi) {
    call_once(initEdgeFlag, initEdge);
    double phiHalf = phi / 2.0;
    double sinPhiHalf = sin(phiHalf);
    double cosPhiHalf = cos(phiHalf);

    QMDDEdge rxyEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(cosPhiHalf, nullptr)}
    }));

    QMDDEdge rxyEdge2 = QMDDEdge(-i * sinPhiHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge rxyEdge3 = QMDDEdge(-i * sinPhiHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge rxyEdge4 = QMDDEdge(cosPhiHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(1.0 * mathUtils::sec(phiHalf), nullptr)}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {rxyEdge1, rxyEdge2},
        {rxyEdge3, rxyEdge4}
    })));
}

QMDDGate gate::SquareSWAP() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge squareSWAPEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge((1.0 + i) / 2.0, nullptr)}
    }));

    QMDDEdge squareSWAPEdge2 = QMDDEdge((1.0 - i) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge squareSWAPEdge3 = QMDDEdge((1.0 - i) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge squareSWAPEdge4 = QMDDEdge((1.0 + i) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(1.0 - i, nullptr)}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {squareSWAPEdge1, squareSWAPEdge2},
        {squareSWAPEdge3, squareSWAPEdge4}
    })));
}

QMDDGate gate::SquareiSWAP() {
    call_once(initEdgeFlag, initEdge);

    QMDDEdge squareiSWAPEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(1.0 / M_SQRT2, nullptr)}
    }));

    QMDDEdge squareiSWAPEdge2 = QMDDEdge(i / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge squareiSWAPEdge3 = QMDDEdge(i / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge squareiSWAPEdge4 = QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(M_SQRT2, nullptr)}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {squareiSWAPEdge1, squareiSWAPEdge2},
        {squareiSWAPEdge3, squareiSWAPEdge4}
    })));
}

QMDDGate gate::SWAPalpha(double alpha) {
    call_once(initEdgeFlag, initEdge);
    complex<double> expIPiAlpha = exp(i * M_PI * alpha);

    QMDDEdge SWAPalphaEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge((1.0 + expIPiAlpha) / 2.0, nullptr)}
    }));

    QMDDEdge SWAPalphaEdge2 = QMDDEdge((1.0 - expIPiAlpha) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge SWAPalphaEdge3 = QMDDEdge((1.0 - expIPiAlpha) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge SWAPalphaEdge4 = QMDDEdge((1.0 + expIPiAlpha) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(2.0 / (1.0 + expIPiAlpha), nullptr)}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {SWAPalphaEdge1, SWAPalphaEdge2},
        {SWAPalphaEdge3, SWAPalphaEdge4}
    })));
}

QMDDGate gate::FREDKIN() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge fredkinEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fredkinEdge1, edgeZero},
        {edgeZero, gate::SWAP().getInitialEdge()}
    })));
}

QMDDGate gate::U(double theta, double phi, double lambda) {
    call_once(initEdgeFlag, initEdge);
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-exp(i * lambda) * tanThetaHalf, nullptr)},
        {QMDDEdge(exp(i * phi) * tanThetaHalf, nullptr), QMDDEdge(exp(i * (lambda + phi)), nullptr)}
    })));
}

QMDDGate gate::U1(double theta) {
    call_once(initEdgeFlag, initEdge);

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, exp(i * theta)}
    })));
}

QMDDGate gate::U2(double phi, double lamda) {
    call_once(initEdgeFlag, initEdge);

    return QMDDGate(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-exp(i * lamda), nullptr)},
        {QMDDEdge(exp(i * phi), nullptr), QMDDEdge(exp(i * (lamda + phi)), nullptr)}
    })));
}

QMDDGate gate::U3(double theta, double phi, double lamda) {
    call_once(initEdgeFlag, initEdge);
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-exp(i * lamda) * tanThetaHalf, nullptr)},
        {QMDDEdge(exp(i * phi) * tanThetaHalf, nullptr), QMDDEdge(exp(i * (lamda + phi)), nullptr)}
    })));
}

QMDDGate gate::BARENCO(double alpha, double phi, double theta) {
    call_once(initEdgeFlag, initEdge);
    double tanTheta = tan(theta);

    QMDDEdge barencoEdge1 = QMDDEdge(exp(i * alpha) * cos(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i * exp(-i * phi) * tanTheta, nullptr)},
        {QMDDEdge(-i * exp(i * phi) * tanTheta, nullptr), edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, barencoEdge1}
    })));
}

QMDDGate gate::B() {
    call_once(initEdgeFlag, initEdge);
    double oneEighthPi = M_PI / 8.0;
    double threeEighthsPi = 3.0 * oneEighthPi;
    double sinThreeEighthsPi = sin(threeEighthsPi);
    double cosThreeEighthsPi = cos(threeEighthsPi);
    double cosOneEighthPi = cos(oneEighthPi);

    QMDDEdge bEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(cosThreeEighthsPi * mathUtils::sec(oneEighthPi), nullptr)}
    }));

    QMDDEdge bEdge2 = QMDDEdge(i * tan(oneEighthPi), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(sinThreeEighthsPi * mathUtils::csc(oneEighthPi), nullptr), edgeZero}
    }));

    QMDDEdge bEdge3 = QMDDEdge(i * sinThreeEighthsPi / cosOneEighthPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(sin(oneEighthPi) * mathUtils::csc(threeEighthsPi), nullptr), edgeZero}
    }));

    QMDDEdge bEdge4 = QMDDEdge(cosThreeEighthsPi / cosOneEighthPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(cosOneEighthPi * mathUtils::sec(threeEighthsPi), nullptr)}
    }));

    return QMDDGate(QMDDEdge(cosOneEighthPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        { bEdge1, bEdge2},
        { bEdge3, bEdge4}
    })));
}

QMDDGate gate::CSX() {
    call_once(initEdgeFlag, initEdge);
    complex<double> expMinusIPiHalf = exp(i * M_PI_4);

    QMDDEdge csxEdge1 = QMDDEdge(exp(i * M_PI_4), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(expMinusIPiHalf, nullptr)},
        {QMDDEdge(expMinusIPiHalf, nullptr), edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, csxEdge1}
    })));
}

QMDDGate gate::N(double a, double b, double c) {
    call_once(initEdgeFlag, initEdge);
    double cosAPlusB = cos(a + b);
    double cosAMinusB = cos(a - b);
    double secAMinusB = mathUtils::sec(a - b);
    complex<double> exp2IC = exp(2.0 * i * c);
    complex<double> expMinus2IC = exp(-2.0 * i * c);
    QMDDEdge nEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(expMinus2IC * cosAPlusB * secAMinusB, nullptr)}
    }));

    QMDDEdge nEdge2 = QMDDEdge(i * tan(a - b), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(expMinus2IC * sin(a + b) * mathUtils::csc(a - b), nullptr), edgeZero}
    }));

    QMDDEdge nEdge3 = QMDDEdge(i * expMinus2IC * sin(a + b) * secAMinusB, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(exp2IC * sin(a - b) * mathUtils::csc(a + b), nullptr), edgeZero}
    }));

    QMDDEdge nEdge4 = QMDDEdge(expMinus2IC * cosAPlusB * secAMinusB, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp2IC * cosAMinusB * mathUtils::sec(a + b), nullptr)}
    }));

    return QMDDGate(QMDDEdge(exp(i * c) * cosAMinusB, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {nEdge1, nEdge2},
        {nEdge3, nEdge4}
    })));
}

QMDDGate gate::DB() {
    call_once(initEdgeFlag, initEdge);
    double threeEighthsPi = 3.0 * M_PI / 8.0;
    double sinThreeEighthsPi = sin(threeEighthsPi);
    double cosThreeEighthsPi = cos(threeEighthsPi);

    QMDDEdge dbEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(cosThreeEighthsPi, nullptr)}
    }));

    QMDDEdge dbEdge2 = QMDDEdge(-i * sinThreeEighthsPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge dbEdge3 = QMDDEdge(-i * sinThreeEighthsPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge dbEdge4 = QMDDEdge(cosThreeEighthsPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(1.0 * mathUtils::sec(threeEighthsPi), nullptr)}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dbEdge1, dbEdge2},
        {dbEdge3, dbEdge4}
    })));
}

QMDDGate gate::ECR() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge ecrEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(i, nullptr)},
        {QMDDEdge(i, nullptr), edgeOne}
    }));

    QMDDEdge ecrEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i, nullptr)},
        {QMDDEdge(-i, nullptr), edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, ecrEdge1},
        {ecrEdge2, edgeZero}
    })));
}

QMDDGate gate::fSim(double theta, double phi) {
    call_once(initEdgeFlag, initEdge);
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);

    QMDDEdge fSimEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(cosTheta, nullptr)}
    }));

    QMDDEdge fSimEdge2 = QMDDEdge(-i * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge fSimEdge3 = QMDDEdge(-i * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge fSimEdge4 = QMDDEdge(cosTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * phi) * mathUtils::sec(theta), nullptr)}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fSimEdge1, fSimEdge2},
        {fSimEdge3, fSimEdge4}
    })));
}

QMDDGate gate::G(double theta) {
    call_once(initEdgeFlag, initEdge);
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);

    QMDDEdge gEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(cosTheta, nullptr)}
    }));

    QMDDEdge gEdge2 = QMDDEdge(-sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge gEdge3 = QMDDEdge(sinTheta,make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge gEdge4 = QMDDEdge(cosTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(1.0 * mathUtils::sec(theta), nullptr)}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gEdge1, gEdge2},
        { gEdge3, gEdge4}
    })));
}

QMDDGate gate::M() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge mEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(i, nullptr)},
        {edgeZero, edgeZero}
    }));

    QMDDEdge mEdge2 = QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, QMDDEdge(-i, nullptr)}
    }));

    QMDDEdge mEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, QMDDEdge(-i, nullptr)}
    }));

    QMDDEdge mEdge4 = QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(i, nullptr)},
        {edgeZero, edgeZero}
    }));

    return QMDDGate(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {mEdge1, mEdge2},
        {mEdge3, mEdge4}
    })));
}

QMDDGate gate::syc() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge sycEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge sycEdge2 = QMDDEdge(-i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge sycEdge3 =QMDDEdge(-i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge sycEdge4 = QMDDEdge(exp(-i * M_PI / 6.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {sycEdge1, sycEdge2},
        {sycEdge3, sycEdge4}
    })));
}

QMDDGate gate::CZS(double theta, double phi, double gamma) {
    call_once(initEdgeFlag, initEdge);
    double sinTheta = sin(theta);
    double sinThetaHalf = sin(theta / 2.0);
    double cosThetaHalf = cos(theta / 2.0);
    double powSinThetaHalf = std::pow(sinThetaHalf, 2);
    double powCosThetaHalf = std::pow(cosThetaHalf, 2);
    complex<double> expIGamma = exp(i * gamma);
    complex<double> expIPhi = exp(i * phi);
    complex<double> expMinusIPhi = exp(-i * phi);

    QMDDEdge czsEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(-expIGamma * powSinThetaHalf + powCosThetaHalf, nullptr)}
    }));

    QMDDEdge czsEdge2 = QMDDEdge((1.0 + expIGamma) / 2.0 * expMinusIPhi * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge czsEdge3 = QMDDEdge((1.0 + expIGamma) / 2.0 * expIPhi * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge czsEdge4 = QMDDEdge(-expIGamma * powCosThetaHalf + powSinThetaHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(-expIGamma / (-expIGamma * powCosThetaHalf + powSinThetaHalf), nullptr)}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {czsEdge1, czsEdge2},
        {czsEdge3, czsEdge4}
    })));
}

QMDDGate gate::D(double theta) {
    call_once(initEdgeFlag, initEdge);
    double tanTheta = tan(theta);
    QMDDEdge dEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    QMDDEdge dEdge2 =  QMDDEdge(i * cos(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i * tanTheta, nullptr)},
        {QMDDEdge(-i * tanTheta, nullptr), edgeOne}
    }));

    QMDDEdge dEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, dEdge2}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dEdge1, edgeZero},
        {edgeZero, dEdge3}
    })));
}

QMDDGate gate::RCCX() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge rccxEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    QMDDEdge rccxEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::Z().getInitialEdge(), edgeZero},
        {edgeZero, gate::X().getInitialEdge()}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {rccxEdge1, edgeZero},
        {edgeZero, rccxEdge2}
    })));
}

QMDDGate gate::PG() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge pgEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    QMDDEdge pgEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, gate::X().getInitialEdge()},
        {gate::I().getInitialEdge(), edgeZero}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {pgEdge1, edgeZero},
        {edgeZero,  pgEdge2}
    })));
}

QMDDGate gate::Toff() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge toffEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {toffEdge1, edgeZero},
        {edgeZero, gate::CX1().getInitialEdge()}
    })));
}

QMDDGate gate::fFredkin() {
    call_once(initEdgeFlag, initEdge);
    QMDDEdge fFredkinEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    QMDDEdge fFredkinEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge fFredkinEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge fFredkinEdge4 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge fFredkinEdge5 = QMDDEdge(-1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    QMDDEdge fFredkinEdge6 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fFredkinEdge2, fFredkinEdge3},
        {fFredkinEdge4, fFredkinEdge5}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fFredkinEdge1, edgeZero},
        {edgeZero, fFredkinEdge6}
    })));
}


// matrix Rotate(const ex &k){
//     return matrix{
//         {1, 0},
//         {0, exp((2 * Pi * I) / pow(2, k))}
//     };
// }
