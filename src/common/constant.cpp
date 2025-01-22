#include "constant.hpp"

#include "../models/state.hpp"
#include "../models/gate.hpp"
#include "mathUtils.hpp"
#include <iostream>

complex<double> i;
QMDDEdge edgeZero;
QMDDEdge edgeOne;
QMDDEdge identityEdge;

QMDDEdge braketZero;
QMDDEdge braketOne;
once_flag initEdgeFlag;
once_flag initExtendedEdgeFlag;

void initEdge() {

    i = complex<double>(.0, 1.0);
    edgeZero = QMDDEdge(.0, nullptr);
    edgeOne = QMDDEdge(1.0, nullptr);
}

void initExtendedEdge() {

    identityEdge = gate::I().getInitialEdge();
    braketZero = mathUtils::mul(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge(), 3);
    braketOne = mathUtils::mul(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge(), 3);
}