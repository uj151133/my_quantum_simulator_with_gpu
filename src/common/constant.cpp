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
    edgeZero = QMDDEdge(.0, 0);
    edgeOne = QMDDEdge(1.0, 0);
}

void initExtendedEdge() {
    identityEdge = gate::I().getInitialEdge();
    braketZero = mathUtils::dyad(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge());
    braketOne = mathUtils::dyad(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge());
}