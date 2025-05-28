#include "bench317tc.hpp"

void bench317tc() {
    QMDDState ket_0 = state::Ket0();
    QuantumCircuit circuit(3);
    circuit.addX({2});
    circuit.addCX(0, 2);
    circuit.addCX(2, 1);
    vector<int> controlIndexes = {1, 2};
    circuit.addToff(controlIndexes, 0);
    controlIndexes = {0, 1};
    circuit.addToff(controlIndexes, 2);
    circuit.addCX(1, 2);
    circuit.simulate();
}