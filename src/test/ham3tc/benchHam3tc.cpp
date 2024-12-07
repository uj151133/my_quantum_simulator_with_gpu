#include "benchHam3tc.hpp"

void benchHam3tc() {
    QMDDState ket_0 = state::Ket0();
    QMDDEdge firstEdge = mathUtils::kronParallel(ket_0.getInitialEdge(), ket_0.getInitialEdge());
    firstEdge = mathUtils::kronParallel(firstEdge, ket_0.getInitialEdge());
    QuantumCircuit circuit(3, QMDDState(firstEdge));
    vector<int> controlIndexes = {1, 2};
    circuit.addToff(controlIndexes, 0);
    circuit.addCX(2, 1);
    circuit.addCX(1, 2);
    circuit.addCX(0, 2);
    circuit.addCX(2, 1);
    circuit.execute();
}