#include "shor.hpp"

void shor(size_t N) {
    size_t m = N == 0 ? 1 : static_cast<size_t>(ceil(log2(N + 1)));
    size_t n = 2 * m;

    QMDDState initialState = state::Ket0();

    for (int _ = 0; _ < m - 1; _++) {
        initialState = mathUtils::kron(state::Ket0().getInitialEdge(), initialState.getInitialEdge());
    }
    initialState = mathUtils::kron(state::Ket1().getInitialEdge(), initialState.getInitialEdge());
    for (int _ = 0; _ < n; _++) {
        initialState = mathUtils::kron(state::Ket0().getInitialEdge(), initialState.getInitialEdge());
    }

    QuantumCircuit circuit(n + m, initialState);

    circuit.setRegister(0, n);
    circuit.setRegister(1, m);

    for (size_t i = 0; i < n; i++) {
        circuit.addH(circuit.quantumRegister[0][i]);
    }
    
    circuit.simulate();

}