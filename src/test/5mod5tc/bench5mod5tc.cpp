#include "bench5mod5tc.hpp"

void bench5mod5tc() {
    QMDDState ket_0 = state::Ket0();
    QMDDEdge firstEdge = mathUtils::kron(ket_0.getInitialEdge(), ket_0.getInitialEdge());
    firstEdge = mathUtils::kron(firstEdge, firstEdge);
    firstEdge = mathUtils::kron(firstEdge, ket_0.getInitialEdge());
    firstEdge = mathUtils::kron(firstEdge, state::Ket1().getInitialEdge());
    QuantumCircuit circuit(6, QMDDState(firstEdge));
    vector<int> controlIndexes = {0, 1, 2, 3, 4};
    circuit.addToff(controlIndexes, 5);
    controlIndexes = {0, 1, 3, 4};
    circuit.addToff(controlIndexes, 5);
    controlIndexes = {3, 4};
    circuit.addToff(controlIndexes, 5);
    circuit.addCX(3, 5);
    controlIndexes = {2, 3};
    circuit.addToff(controlIndexes, 5);
    circuit.addCX(2, 5);
    controlIndexes = {1, 2};
    circuit.addToff(controlIndexes, 5);
    circuit.addCX(1, 5);
    controlIndexes = {0, 1, 2, 4};
    circuit.addToff(controlIndexes, 5);
    controlIndexes = {0, 4};
    circuit.addToff(controlIndexes, 5);
    circuit.addCX(4, 5);
    controlIndexes = {0, 1, 4};
    circuit.addToff(controlIndexes, 5);
    controlIndexes = {1, 4};
    circuit.addToff(controlIndexes, 5);
    controlIndexes = {0, 2, 4};
    circuit.addToff(controlIndexes, 5);
    circuit.addCX(0, 5);
    controlIndexes = {0, 3};
    circuit.addToff(controlIndexes, 5);
    controlIndexes = {0, 1};
    circuit.addToff(controlIndexes, 5);
    circuit.simulate();
}