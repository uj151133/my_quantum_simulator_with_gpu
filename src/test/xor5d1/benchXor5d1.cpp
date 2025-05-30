#include "benchXor5d1.hpp"

void benchXor5d1() {
    QMDDState ket_0 = state::Ket0();
    QuantumCircuit circuit(5);
    circuit.addCX(0, 1);
    circuit.addCX(1, 2);
    circuit.addCX(2, 3);
    circuit.addCX(3, 4);
    circuit.simulate();
}