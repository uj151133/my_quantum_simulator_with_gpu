#include "QFT.hpp"

void QFT(size_t numQubits) {

    QuantumCircuit circuit(numQubits);

    circuit.addQFT();

    circuit.simulate();
}