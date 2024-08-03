// OpenQASM 2.0
include "qelib1.inc"; // 標準ゲートライブラリのインクルード


// Define quantum and classical registers
qreg q[2];
creg c[2];

// Apply a Hadamard gate to the first qubit
h q[0];

// Apply a CNOT gate with the first qubit as control and the second qubit as target
cx q[0], q[1];

// Measure both qubits
measure q[0] -> c[0];
measure q[1] -> c[1];
