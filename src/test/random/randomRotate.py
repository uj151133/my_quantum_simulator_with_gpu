from qiskit import QuantumCircuit
from qiskit_aer import Aer
import numpy as np
from numpy.random import default_rng

def random_rotate(num_qubits: int, num_gates: int = 200):
    circuit = QuantumCircuit(num_qubits)

    rng = default_rng()

    for _ in range(num_gates):
        qubit = rng.integers(0, num_qubits)
        phase = rng.uniform(0, 2 * np.pi)
        gate_type = rng.integers(0, 3)

        if gate_type == 0:
            circuit.rx(phase, qubit)
        elif gate_type == 1:
            circuit.ry(phase, qubit)
        else:
            circuit.rz(phase, qubit)

    backend = Aer.get_backend('statevector_simulator')
    job = backend.run(circuit)
    result = job.result()

    return circuit, result

if __name__ == "__main__":
    circuit, result = random_rotate(num_qubits=100)
    print(circuit)