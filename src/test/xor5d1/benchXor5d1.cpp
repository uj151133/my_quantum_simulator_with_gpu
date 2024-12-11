#include "benchXor5d1.hpp"

void benchXor5d1() {
    QMDDState ket_0 = state::Ket0();
    QMDDEdge firstEdge = mathUtils::kron(ket_0.getInitialEdge(), ket_0.getInitialEdge());
    firstEdge = mathUtils::kron(firstEdge, firstEdge);
    firstEdge = mathUtils::kron(firstEdge, ket_0.getInitialEdge());
    QuantumCircuit circuit(5, QMDDState(firstEdge));
    circuit.addCX(0, 1);
    circuit.addCX(1, 2);
    circuit.addCX(2, 3);
    circuit.addCX(3, 4);
    circuit.execute();
}