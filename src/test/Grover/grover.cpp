#include "grover.hpp"

void grover(size_t numQubits, size_t omega) {

    QuantumCircuit circuit(numQubits);

    circuit.addAllH();

    int times = max(1, int(sqrt(numQubits) * M_PI_4));
    for (size_t i = 0; i < times; i++) {
        circuit.addOracle(omega);
        circuit.addIAM();
    }

    circuit.execute();
}