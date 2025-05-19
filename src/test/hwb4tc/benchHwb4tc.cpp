#include "benchHwb4tc.hpp"

void benchHwb4tc() {
    QMDDState ket_0 = state::Ket0();
    QMDDEdge firstEdge = mathUtils::kron(ket_0.getInitialEdge(), ket_0.getInitialEdge());
    firstEdge = mathUtils::kron(firstEdge, firstEdge);
    QuantumCircuit circuit(4, QMDDState(firstEdge));
    circuit.addCX(3, 1);
    circuit.addCX(2, 3);
    circuit.addCX(3, 2);
    vector<int> controlIndexes = {0, 3};
    circuit.addToff(controlIndexes, 2);
    controlIndexes = {2, 3};
    circuit.addToff(controlIndexes, 0);
    controlIndexes = {1, 2, 3};
    circuit.addToff(controlIndexes, 0);
    controlIndexes = {1, 3};
    circuit.addToff(controlIndexes, 0);
    controlIndexes = {0, 1, 2};
    circuit.addToff(controlIndexes, 3);
    controlIndexes = {0, 3};
    circuit.addToff(controlIndexes, 2);
    circuit.addCX(3, 0);
    controlIndexes = {0, 2};
    circuit.addToff(controlIndexes, 3);
    controlIndexes = {0, 1};
    circuit.addToff(controlIndexes, 3);
    circuit.addCX(3, 0);
    circuit.addCX(1, 2);
    circuit.addCX(2, 1);
    circuit.addCX(0, 1);
    circuit.addCX(1, 0);
    circuit.simulate();
}