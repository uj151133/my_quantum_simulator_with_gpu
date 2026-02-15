#include "bigBang.hpp"

complex<double> i;
QMDDEdge edgeZero;
QMDDEdge edgeOne;
QMDDEdge identityEdge;

QMDDEdge braketZero;
QMDDEdge braketOne;

void birth() {

    /* global resources initialization */
    PARAMETER.load();
    (void)threadPool;
    UniqueTable::getInstance();
    OperationCacheClient::getInstance();

    /* edge initialization */
    i = complex<double>(.0, 1.0);
    edgeZero = QMDDEdge(.0, 0);
    edgeOne = QMDDEdge(1.0, 0);

    /* gate initialization */
    identityEdge = gate::I().getInitialEdge();
    braketZero = mathUtils::dyad(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge());
    braketOne = mathUtils::dyad(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge());
}