#include "benchHam3tc.hpp"

void benchHam3tc() {
    QuantumCircuit circuit(3);
    vector<int> controlIndexes = {1, 2};
    circuit.addToff(controlIndexes, 0);
    circuit.addCX(2, 1);
    circuit.addCX(1, 2);
    circuit.addCX(0, 2);
    circuit.addCX(2, 1);
    circuit.simulate();
}