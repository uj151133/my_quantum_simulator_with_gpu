#include "benchHwb4tc.hpp"

void benchHwb4tc() {
    QuantumCircuit circuit(4);
    circuit.addCX(3, 1);
    circuit.addCX(2, 3);
    circuit.addCX(3, 2);
    array<int, 2> controlToff = {0, 3};
    circuit.addToff(controlToff, 2);
    controlToff = {2, 3};
    circuit.addToff(controlToff, 0);
    vector<int> controlMCT = {1, 2, 3};
    circuit.addMCT(controlMCT, 0);
    controlToff = {1, 3};
    circuit.addToff(controlToff, 0);
    controlMCT = {0, 1, 2};
    circuit.addMCT(controlMCT, 3);
    controlToff = {0, 3};
    circuit.addToff(controlToff, 2);
    circuit.addCX(3, 0);
    controlToff = {0, 2};
    circuit.addToff(controlToff, 3);
    controlToff = {0, 1};
    circuit.addToff(controlToff, 3);
    circuit.addCX(3, 0);
    circuit.addCX(1, 2);
    circuit.addCX(2, 1);
    circuit.addCX(0, 1);
    circuit.addCX(1, 0);
    circuit.simulate();
}