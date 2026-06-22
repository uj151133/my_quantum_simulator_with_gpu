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
        case Type::CY: os << "CY"; break;
        case Type::CZ: os << "CZ"; break;
        case Type::SWAP: os << "SWAP"; break;
        case Type::P: os << "P"; break;
        case Type::T: os << "T"; break;
        case Type::Tdg: os << "Tdg"; break;
        case Type::CP: os << "CP"; break;
        case Type::CS: os << "CS"; break;
        case Type::CH: os << "CH"; break;
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
        case Type::CRx: os << "CRx"; break;
        case Type::CRy: os << "CRy"; break;
        case Type::CRz: os << "CRz"; break;
        case Type::CU: os << "CU"; break;
        case Type::Toff: os << "Toff"; break;
        case Type::MCT: os << "MCT"; break;
        case Type::Other: os << "Other"; break;
        case Type::VOID: os << "⛓️"; break;
        case Type::ANKER: os << "⚓️"; break;
        case Type::BAN: os << "🚫"; break;
        case Type::JOKER: os << "🃏"; break;
        default: os << "Unknown"; break;
    }
    return os;
}

string toString(Type type) {
    switch (type) {
        case Type::I: return "I";
        case Type::Ph: return "Ph";
        case Type::X: return "X";
        case Type::Y: return "Y";
        case Type::Z: return "Z";
        case Type::S: return "S";
        case Type::Sdg: return "Sdg";
        case Type::V: return "V";
        case Type::Vdg: return "Vdg";
        case Type::H: return "H";
        case Type::CX: return "CX";
        case Type::varCX: return "varCX";
        case Type::CY: return "CY";
        case Type::CZ: return "CZ";
        case Type::SWAP: return "SWAP";
        case Type::P: return "P";
        case Type::T: return "T";
        case Type::Tdg: return "Tdg";
        case Type::CP: return "CP";
        case Type::CS: return "CS";
        case Type::CH: return "CH";
        case Type::R: return "R";
        case Type::Rx: return "Rx";
        case Type::Ry: return "Ry";
        case Type::Rz: return "Rz";
        case Type::Rxx: return "Rxx";
        case Type::Ryy: return "Ryy";
        case Type::Rzz: return "Rzz";
        case Type::Rxy: return "Rxy";
        case Type::U: return "U";
        case Type::U1: return "U1";
        case Type::U2: return "U2";
        case Type::U3: return "U3";
        case Type::CRx: return "CRx";
        case Type::CRy: return "CRy";
        case Type::CRz: return "CRz";
        case Type::CU: return "CU";
        case Type::Toff: return "Toff";
        case Type::Other: return "Other";
        case Type::VOID: return "VOID";
        case Type::BAN: return "BAN";
        case Type::JOKER: return "JOKER";
        default: return "Unknown";
    }
}


QMDDSuite gate::I() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeOne}
    }));
}

QMDDSuite gate::Ph(double delta) {
    return QMDDSuite(mathUtils::toLogPolar(exp(i * delta)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeOne}
    }));
}

QMDDSuite gate::X() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeOne, edgeZero}
    }));
}

// const QMDDSuite gate::PLUS_X_GATE = [] {
//     complex<double> plusXWeight = 1 / M_SQRT2;
//     QMDDEdge plusXNode = make_shared<QMDDNode>(4);

//     plusXNode->children[0] = QMDDEdge(1);
//     plusXNode->children[1] = QMDDEdge(mathUtils::toLogPolar(i));
//     plusXNode->children[2] = QMDDEdge(mathUtils::toLogPolar(i));
//     plusXNode->children[3] = QMDDEdge(1);

//     QMDDEdge plusXEdge(plusXWeight, plusXNode);
//     return QMDDSuite(plusXEdge);
// }();

// const QMDDSuite gate::MINUS_X_GATE = [] {
//     complex<double> minusXWeight = 1 / M_SQRT2;
//     QMDDEdge minusXNode = make_shared<QMDDNode>(4);

//     minusXNode->children[0] = QMDDEdge(1);
//     minusXNode->children[1] = QMDDEdge(mathUtils::toLogPolar(-i));
//     minusXNode->children[2] = QMDDEdge(mathUtils::toLogPolar(-i));
//     minusXNode->children[3] = QMDDEdge(1);

//     QMDDEdge minusXEdge(minusXWeight, minusXNode);
//     return QMDDSuite(minusXEdge);
// }();

QMDDSuite gate::Y() {
    return QMDDSuite(mathUtils::toLogPolar(-i), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(mathUtils::toLogPolar(-1.0)), edgeZero}
    }));
}

// QMDDSuite createPlusYGate() {
//     complex<double> plusYWeight = 1 / M_SQRT2;
//     QMDDNode* plusYNode = new QMDDNode(4);

//     plusYNode->children[0] = QMDDEdge(1);
//     plusYNode->children[1] = QMDDEdge(1);
//     plusYNode->children[2] = QMDDEdge(-1);
//     plusYNode->children[3] = QMDDEdge(1);

//     QMDDEdge plusYEdge(plusYWeight, plusYNode);
//     return QMDDSuite(plusYEdge);
// }

// QMDDSuite createMinusYGate() {
//     complex<double> minusYWeight = 1 / M_SQRT2;
//     QMDDNode* minusYNode = new QMDDNode(4);

//     minusYNode->children[0] = QMDDEdge(1);
//     minusYNode->children[1] = QMDDEdge(-1);
//     minusYNode->children[2] = QMDDEdge(1);
//     minusYNode->children[3] = QMDDEdge(1);

//     QMDDEdge minusYEdge(minusYWeight, minusYNode);
//     return QMDDSuite(minusYEdge);
// }
QMDDSuite gate::Z() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(-1.0))}
    }));
}

QMDDSuite gate::S() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(i))}
    }));
}

QMDDSuite gate::Sdg() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(-i))}
    }));
}

QMDDSuite gate::V() {
    QMDDEdge vEdge = QMDDEdge(mathUtils::toLogPolar(i));

    return QMDDSuite(mathUtils::toLogPolar(0.5 + 0.5 * i), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, vEdge},
        {vEdge, edgeOne}
    }));
}

QMDDSuite gate::Vdg() {
    QMDDEdge vdgEdge = QMDDEdge(mathUtils::toLogPolar(i));

    return QMDDSuite(mathUtils::toLogPolar(0.5 - 0.5 * i), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, vdgEdge},
        {vdgEdge, edgeOne}
    }));
}

QMDDSuite gate::H() {
    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeOne},
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-1.0))}
    }));
}

QMDDSuite gate::CX1() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge() , edgeZero},
        {edgeZero, gate::X().getInitialEdge()}
    }));
}

QMDDSuite gate::CX2() {
    QMDDEdge cx2Edge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge cx2Edge2 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {cx2Edge1, cx2Edge2},
        {cx2Edge2, cx2Edge1}
    }));
}

QMDDSuite gate::varCX() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::X().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));
}

QMDDSuite gate::CZ() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::Z().getInitialEdge()}
    }));
}

QMDDSuite gate::DCNOT() {
    QMDDEdge dcnotEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge dcnotEdge2 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge dcnotEdge3 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    QMDDEdge dcnotEdge4 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dcnotEdge1, dcnotEdge2},
        {dcnotEdge3, dcnotEdge4}
    }));
}

QMDDSuite gate::SWAP() {
    QMDDEdge swapEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge swapEdge2 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge swapEdge3 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge swapEdge4 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {swapEdge1, swapEdge2},
        {swapEdge3, swapEdge4}
    }));
}

QMDDSuite gate::iSWAP() {
    QMDDEdge iswapEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge iswapEdge2 = QMDDEdge(mathUtils::toLogPolar(i), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge iswapEdge3 = QMDDEdge(mathUtils::toLogPolar(i), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge iswapEdge4 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {iswapEdge1, iswapEdge2},
        {iswapEdge3, iswapEdge4}
    }));
}

QMDDSuite gate::P(double phi) {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(exp(i * phi)))}
    }));
}

QMDDSuite gate::T() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(exp(i * M_PI_4)))}
    }));
}

QMDDSuite gate::Tdg() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(exp(-i * M_PI_4)))}
    }));
}

QMDDSuite gate::CP(double phi) {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::P(phi).getInitialEdge()}
    }));
}

QMDDSuite gate::CS() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::S().getInitialEdge()}
    }));
}

QMDDSuite gate::R(double theta, double phi) {
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDSuite(mathUtils::toLogPolar(cos(thetaHalf)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-i * exp(-i * phi) * tanThetaHalf))},
        {QMDDEdge(mathUtils::toLogPolar(-i * exp(i * phi) * tanThetaHalf)), edgeOne}
    }));
}

QMDDSuite gate::Rx(double theta) {
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDSuite(mathUtils::toLogPolar(cos(thetaHalf)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-i * tanThetaHalf))},
        {QMDDEdge(mathUtils::toLogPolar(-i * tanThetaHalf)), edgeOne}
    }));
}

QMDDSuite gate::Ry(double theta) {
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDSuite(mathUtils::toLogPolar(cos(thetaHalf)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-tanThetaHalf))},
        {QMDDEdge(mathUtils::toLogPolar(tanThetaHalf)), edgeOne}
    }));
}

QMDDSuite gate::Rz(double theta) {
    return QMDDSuite(mathUtils::toLogPolar(exp(-i * theta / 2.0)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(exp(i * theta)))}
    }));
}

QMDDSuite gate::Rk(int k) {
    complex<double> theta = 2 * M_PI * i / pow(2, k);
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(exp(theta)))}
    }));
}

QMDDSuite gate::Rxx(double phi) {
    double phiHalf = phi / 2.0;
    double tanPhiHalf = tan(phiHalf);

    return QMDDSuite(mathUtils::toLogPolar(cos(phiHalf)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), QMDDEdge(mathUtils::toLogPolar(-i * tanPhiHalf), make_shared<QMDDNode>(*get<shared_ptr<QMDDNode>>(gate::X().getInitialEdge().getSon())))},
        {QMDDEdge(mathUtils::toLogPolar(-i * tanPhiHalf), make_shared<QMDDNode>(*get<shared_ptr<QMDDNode>>(gate::X().getInitialEdge().getSon()))), gate::I().getInitialEdge()}
    }));
}

QMDDSuite gate::Ryy(double phi) {
    double phiHalf = phi / 2.0;
    double tanPhiHalf = tan(phiHalf);

    return QMDDSuite(mathUtils::toLogPolar(cos(phiHalf)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), QMDDEdge(mathUtils::toLogPolar(i * tanPhiHalf), make_shared<QMDDNode>(*get<shared_ptr<QMDDNode>>(gate::Y().getInitialEdge().getSon())))},
        {QMDDEdge(mathUtils::toLogPolar(-i * tanPhiHalf), make_shared<QMDDNode>(*get<shared_ptr<QMDDNode>>(gate::Y().getInitialEdge().getSon()))), gate::I().getInitialEdge()}
    }));
}

QMDDSuite gate::Rzz(double phi) {
    return QMDDSuite(mathUtils::toLogPolar(exp(-i * phi / 2.0)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::P(phi).getInitialEdge(), edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(exp(i * phi)), make_shared<QMDDNode>(*get<shared_ptr<QMDDNode>>(gate::P(-phi).getInitialEdge().getSon())))}
    }));
}

QMDDSuite gate::Rxy(double phi) {
    double phiHalf = phi / 2.0;
    double sinPhiHalf = sin(phiHalf);
    double cosPhiHalf = cos(phiHalf);

    QMDDEdge rxyEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(cosPhiHalf))}
    }));

    QMDDEdge rxyEdge2 = QMDDEdge(mathUtils::toLogPolar(-i * sinPhiHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge rxyEdge3 = QMDDEdge(mathUtils::toLogPolar(-i * sinPhiHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge rxyEdge4 = QMDDEdge(mathUtils::toLogPolar(cosPhiHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(1.0 * mathUtils::sec(phiHalf)))}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {rxyEdge1, rxyEdge2},
        {rxyEdge3, rxyEdge4}
    }));
}

QMDDSuite gate::SquareSWAP() {
    QMDDEdge squareSWAPEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar((1.0 + i) / 2.0))}
    }));

    QMDDEdge squareSWAPEdge2 = QMDDEdge(mathUtils::toLogPolar((1.0 - i) / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge squareSWAPEdge3 = QMDDEdge(mathUtils::toLogPolar((1.0 - i) / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge squareSWAPEdge4 = QMDDEdge(mathUtils::toLogPolar((1.0 + i) / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(1.0 - i))}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {squareSWAPEdge1, squareSWAPEdge2},
        {squareSWAPEdge3, squareSWAPEdge4}
    }));
}

QMDDSuite gate::SquareiSWAP() {

    QMDDEdge squareiSWAPEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(1.0 / M_SQRT2))}
    }));

    QMDDEdge squareiSWAPEdge2 = QMDDEdge(mathUtils::toLogPolar(i / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge squareiSWAPEdge3 = QMDDEdge(mathUtils::toLogPolar(i / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge squareiSWAPEdge4 = QMDDEdge(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(M_SQRT2))}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {squareiSWAPEdge1, squareiSWAPEdge2},
        {squareiSWAPEdge3, squareiSWAPEdge4}
    }));
}

QMDDSuite gate::SWAPalpha(double alpha) {
    complex<double> expIPiAlpha = exp(i * M_PI * alpha);

    QMDDEdge SWAPalphaEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar((1.0 + expIPiAlpha) / 2.0))}
    }));

    QMDDEdge SWAPalphaEdge2 = QMDDEdge(mathUtils::toLogPolar((1.0 - expIPiAlpha) / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge SWAPalphaEdge3 = QMDDEdge(mathUtils::toLogPolar((1.0 - expIPiAlpha) / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge SWAPalphaEdge4 = QMDDEdge(mathUtils::toLogPolar((1.0 + expIPiAlpha) / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(2.0 / (1.0 + expIPiAlpha)))}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {SWAPalphaEdge1, SWAPalphaEdge2},
        {SWAPalphaEdge3, SWAPalphaEdge4}
    }));
}

QMDDSuite gate::FREDKIN() {
    QMDDEdge fredkinEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fredkinEdge1, edgeZero},
        {edgeZero, gate::SWAP().getInitialEdge()}
    }));
}

QMDDSuite gate::U(double theta, double phi, double lambda) {
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDSuite(mathUtils::toLogPolar(cos(thetaHalf)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-exp(i * lambda) * tanThetaHalf))},
        {QMDDEdge(mathUtils::toLogPolar(exp(i * phi) * tanThetaHalf)), QMDDEdge(mathUtils::toLogPolar(exp(i * (lambda + phi))))}
    }));
}

QMDDSuite gate::U1(double theta) {

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(exp(i * theta)))}
    }));
}

QMDDSuite gate::U2(double phi, double lamda) {

    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-exp(i * lamda)))},
        {QMDDEdge(mathUtils::toLogPolar(exp(i * phi))), QMDDEdge(mathUtils::toLogPolar(exp(i * (lamda + phi))))}
    }));
}

QMDDSuite gate::U3(double theta, double phi, double lamda) {
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDSuite(mathUtils::toLogPolar(cos(thetaHalf)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-exp(i * lamda) * tanThetaHalf))},
        {QMDDEdge(mathUtils::toLogPolar(exp(i * phi) * tanThetaHalf)), QMDDEdge(mathUtils::toLogPolar(exp(i * (lamda + phi))))}
    }));
}

QMDDSuite gate::BARENCO(double alpha, double phi, double theta) {
    double tanTheta = tan(theta);

    QMDDEdge barencoEdge1 = QMDDEdge(mathUtils::toLogPolar(exp(i * alpha) * cos(theta)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-i * exp(-i * phi) * tanTheta))},
        {QMDDEdge(mathUtils::toLogPolar(-i * exp(i * phi) * tanTheta)), edgeOne}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, barencoEdge1}
    }));
}

QMDDSuite gate::B() {
    double oneEighthPi = M_PI / 8.0;
    double threeEighthsPi = 3.0 * oneEighthPi;
    double sinThreeEighthsPi = sin(threeEighthsPi);
    double cosThreeEighthsPi = cos(threeEighthsPi);
    double cosOneEighthPi = cos(oneEighthPi);

    QMDDEdge bEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(cosThreeEighthsPi * mathUtils::sec(oneEighthPi)))}
    }));

    QMDDEdge bEdge2 = QMDDEdge(mathUtils::toLogPolar(i * tan(oneEighthPi)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(mathUtils::toLogPolar(sinThreeEighthsPi * mathUtils::csc(oneEighthPi))), edgeZero}
    }));

    QMDDEdge bEdge3 = QMDDEdge(mathUtils::toLogPolar(i * sinThreeEighthsPi / cosOneEighthPi), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(mathUtils::toLogPolar(sin(oneEighthPi) * mathUtils::csc(threeEighthsPi))), edgeZero}
    }));

    QMDDEdge bEdge4 = QMDDEdge(mathUtils::toLogPolar(cosThreeEighthsPi / cosOneEighthPi), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(cosOneEighthPi * mathUtils::sec(threeEighthsPi)))}
    }));

    return QMDDSuite(mathUtils::toLogPolar(cosOneEighthPi), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        { bEdge1, bEdge2},
        { bEdge3, bEdge4}
    }));
}

QMDDSuite gate::CSX() {
    complex<double> expMinusIPiHalf = exp(i * M_PI_4);

    QMDDEdge csxEdge1 = QMDDEdge(mathUtils::toLogPolar(exp(i * M_PI_4)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(expMinusIPiHalf))},
        {QMDDEdge(mathUtils::toLogPolar(expMinusIPiHalf)), edgeOne}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, csxEdge1}
    }));
}

QMDDSuite gate::N(double a, double b, double c) {
    double cosAPlusB = cos(a + b);
    double cosAMinusB = cos(a - b);
    double secAMinusB = mathUtils::sec(a - b);
    complex<double> exp2IC = exp(2.0 * i * c);
    complex<double> expMinus2IC = exp(-2.0 * i * c);
    QMDDEdge nEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(expMinus2IC * cosAPlusB * secAMinusB))}
    }));

    QMDDEdge nEdge2 = QMDDEdge(mathUtils::toLogPolar(i * tan(a - b)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(mathUtils::toLogPolar(expMinus2IC * sin(a + b) * mathUtils::csc(a - b))), edgeZero}
    }));

    QMDDEdge nEdge3 = QMDDEdge(mathUtils::toLogPolar(i * expMinus2IC * sin(a + b) * secAMinusB), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(mathUtils::toLogPolar(exp2IC * sin(a - b) * mathUtils::csc(a + b))), edgeZero}
    }));

    QMDDEdge nEdge4 = QMDDEdge(mathUtils::toLogPolar(expMinus2IC * cosAPlusB * secAMinusB), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(exp2IC * cosAMinusB * mathUtils::sec(a + b)))}
    }));

    return QMDDSuite(mathUtils::toLogPolar(exp(i * c) * cosAMinusB), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {nEdge1, nEdge2},
        {nEdge3, nEdge4}
    }));
}

QMDDSuite gate::DB() {
    double threeEighthsPi = 3.0 * M_PI / 8.0;
    double sinThreeEighthsPi = sin(threeEighthsPi);
    double cosThreeEighthsPi = cos(threeEighthsPi);

    QMDDEdge dbEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(cosThreeEighthsPi))}
    }));

    QMDDEdge dbEdge2 = QMDDEdge(mathUtils::toLogPolar(-i * sinThreeEighthsPi), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge dbEdge3 = QMDDEdge(mathUtils::toLogPolar(-i * sinThreeEighthsPi), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge dbEdge4 = QMDDEdge(mathUtils::toLogPolar(cosThreeEighthsPi), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(1.0 * mathUtils::sec(threeEighthsPi)))}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dbEdge1, dbEdge2},
        {dbEdge3, dbEdge4}
    }));
}

QMDDSuite gate::ECR() {
    QMDDEdge ecrEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(i))},
        {QMDDEdge(mathUtils::toLogPolar(i)), edgeOne}
    }));

    QMDDEdge ecrEdge2 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-i))},
        {QMDDEdge(mathUtils::toLogPolar(-i)), edgeOne}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, ecrEdge1},
        {ecrEdge2, edgeZero}
    }));
}

QMDDSuite gate::fSim(double theta, double phi) {
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);

    QMDDEdge fSimEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(cosTheta))}
    }));

    QMDDEdge fSimEdge2 = QMDDEdge(mathUtils::toLogPolar(-i * sinTheta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge fSimEdge3 = QMDDEdge(mathUtils::toLogPolar(-i * sinTheta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge fSimEdge4 = QMDDEdge(mathUtils::toLogPolar(cosTheta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(exp(i * phi) * mathUtils::sec(theta)))}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fSimEdge1, fSimEdge2},
        {fSimEdge3, fSimEdge4}
    }));
}

QMDDSuite gate::G(double theta) {
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);

    QMDDEdge gEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(cosTheta))}
    }));

    QMDDEdge gEdge2 = QMDDEdge(mathUtils::toLogPolar(-sinTheta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge gEdge3 = QMDDEdge(mathUtils::toLogPolar(sinTheta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge gEdge4 = QMDDEdge(mathUtils::toLogPolar(cosTheta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(1.0 * mathUtils::sec(theta)))}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gEdge1, gEdge2},
        { gEdge3, gEdge4}
    }));
}

QMDDSuite gate::M() {
    QMDDEdge mEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(i))},
        {edgeZero, edgeZero}
    }));

    QMDDEdge mEdge2 = QMDDEdge(mathUtils::toLogPolar(i), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-i))}
    }));

    QMDDEdge mEdge3 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-i))}
    }));

    QMDDEdge mEdge4 = QMDDEdge(mathUtils::toLogPolar(i), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(i))},
        {edgeZero, edgeZero}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {mEdge1, mEdge2},
        {mEdge3, mEdge4}
    }));
}

QMDDSuite gate::syc() {
    QMDDEdge sycEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge sycEdge2 = QMDDEdge(mathUtils::toLogPolar(-i), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge sycEdge3 = QMDDEdge(mathUtils::toLogPolar(-i), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge sycEdge4 = QMDDEdge(mathUtils::toLogPolar(exp(-i * M_PI / 6.0)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {sycEdge1, sycEdge2},
        {sycEdge3, sycEdge4}
    }));
}

QMDDSuite gate::CZS(double theta, double phi, double gamma) {
    double sinTheta = sin(theta);
    double sinThetaHalf = sin(theta / 2.0);
    double cosThetaHalf = cos(theta / 2.0);
    double powSinThetaHalf = std::pow(sinThetaHalf, 2);
    double powCosThetaHalf = std::pow(cosThetaHalf, 2);
    complex<double> expIGamma = exp(i * gamma);
    complex<double> expIPhi = exp(i * phi);
    complex<double> expMinusIPhi = exp(-i * phi);

    QMDDEdge czsEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(-expIGamma * powSinThetaHalf + powCosThetaHalf))}
    }));

    QMDDEdge czsEdge2 = QMDDEdge(mathUtils::toLogPolar((1.0 + expIGamma) / 2.0 * expMinusIPhi * sinTheta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge czsEdge3 = QMDDEdge(mathUtils::toLogPolar((1.0 + expIGamma) / 2.0 * expIPhi * sinTheta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge czsEdge4 = QMDDEdge(mathUtils::toLogPolar(-expIGamma * powCosThetaHalf + powSinThetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(mathUtils::toLogPolar(-expIGamma / (-expIGamma * powCosThetaHalf + powSinThetaHalf)))}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {czsEdge1, czsEdge2},
        {czsEdge3, czsEdge4}
    }));
}

QMDDSuite gate::D(double theta) {
    double tanTheta = tan(theta);
    QMDDEdge dEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    QMDDEdge dEdge2 =  QMDDEdge(mathUtils::toLogPolar(i * cos(theta)), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-i * tanTheta))},
        {QMDDEdge(mathUtils::toLogPolar(-i * tanTheta)), edgeOne}
    }));

    QMDDEdge dEdge3 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, dEdge2}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dEdge1, edgeZero},
        {edgeZero, dEdge3}
    }));
}

QMDDSuite gate::RCCX() {
    QMDDEdge rccxEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    QMDDEdge rccxEdge2 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::Z().getInitialEdge(), edgeZero},
        {edgeZero, gate::X().getInitialEdge()}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {rccxEdge1, edgeZero},
        {edgeZero, rccxEdge2}
    }));
}

QMDDSuite gate::PG() {
    QMDDEdge pgEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    QMDDEdge pgEdge2 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, gate::X().getInitialEdge()},
        {gate::I().getInitialEdge(), edgeZero}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {pgEdge1, edgeZero},
        {edgeZero,  pgEdge2}
    }));
}

QMDDSuite gate::Toff() {
    QMDDEdge toffEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {toffEdge1, edgeZero},
        {edgeZero, gate::CX1().getInitialEdge()}
    }));
}

QMDDSuite gate::fFredkin() {
    QMDDEdge fFredkinEdge1 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    QMDDEdge fFredkinEdge2 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));

    QMDDEdge fFredkinEdge3 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero}
    }));

    QMDDEdge fFredkinEdge4 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));

    QMDDEdge fFredkinEdge5 = QMDDEdge(mathUtils::toLogPolar(-1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeZero, edgeOne}
    }));

    QMDDEdge fFredkinEdge6 = QMDDEdge(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fFredkinEdge2, fFredkinEdge3},
        {fFredkinEdge4, fFredkinEdge5}
    }));

    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fFredkinEdge1, edgeZero},
        {edgeZero, fFredkinEdge6}
    }));
}


// matrix Rotate(const ex &k){
//     return matrix{
//         {1, 0},
//         {0, exp((2 * Pi * I) / pow(2, k))}
//     };
// }
