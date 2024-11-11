#include "grover.hpp"

void grover() {

    size_t collectNum = 5;
    size_t numQubits = 3;

    QuantumCircuit circuit(numQubits);

    circuit.addAllH();

    for (size_t i = 0; i < int(sqrt(numQubits) * M_PI_4); i++) {
        circuit.addOracle(collectNum);
        circuit.addIAM();
    }
}