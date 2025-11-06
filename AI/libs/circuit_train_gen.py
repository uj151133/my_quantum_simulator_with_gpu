from typing import List
# from qiskit.circuit import QuantumCircuit
from qiskit import QuantumCircuit
import numpy as np

def make_training_circuits(num: int = 20, num_qubits: int = 6, depth: int = 80) -> List[QuantumCircuit]:
    rng = np.random.default_rng(42)
    qcs: List[QuantumCircuit] = []
    for _ in range(num):
        qc = QuantumCircuit(num_qubits)
        for _d in range(depth):
            k = rng.integers(0, 5)
            i = int(rng.integers(0, num_qubits))
            j = int(rng.integers(0, num_qubits))
            if k == 0:
                qc.h(i)
            elif k == 1 and i != j:
                qc.cx(i, j)
            elif k == 2:
                qc.rz(float(rng.random()*2*np.pi- np.pi), i)
            elif k == 3 and i != j:
                qc.cz(i, j)
            else:
                qc.x(i)
        qcs.append(qc)
    return qcs