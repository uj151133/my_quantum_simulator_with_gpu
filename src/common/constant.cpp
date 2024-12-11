#include "constant.hpp"
#include "../models/gate.hpp"
#include "../models/state.hpp"
#include "../common/mathUtils.hpp"
#include <iostream>

complex<double> i(.0, 1.0);
QMDDEdge edgeZero = QMDDEdge(.0, nullptr);
QMDDEdge edgeOne = QMDDEdge(1.0, nullptr);
QMDDEdge identityEdge;  // 非const変数として定義
QMDDEdge braketZero;
QMDDEdge braketOne;
once_flag initEdgeFlag;
once_flag initExtendedEdgeFlag;

void initEdge() {
    // cout << "Initializing constant values" << std::endl;
    i = complex<double>(.0, 1.0);
    edgeZero = QMDDEdge(.0, nullptr);
    edgeOne = QMDDEdge(1.0, nullptr);
}

void initExtendedEdge() {
    // cout << "Initializing extended constant values" << std::endl;
    identityEdge = gate::I().getInitialEdge();
    braketZero = mathUtils::mul(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge());
    braketOne = mathUtils::mul(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge());
}