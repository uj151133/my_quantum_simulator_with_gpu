from typing import List
# from qiskit.circuit import QuantumCircuit
from qiskit import QuantumCircuit, qasm2, qasm3
import numpy as np
from mqt.bench import BenchmarkLevel, get_benchmark
from pathlib import Path

def make_random_circuits(num: int = 3, num_qubits: int = 6, depth: int = 60) -> List[QuantumCircuit]:
    rng = np.random.default_rng(7)
    qcs: List[QuantumCircuit] = []
    for _ in range(num):
        qc = QuantumCircuit(num_qubits)
        for _d in range(depth):
            k = rng.integers(0, 6)
            i = int(rng.integers(0, num_qubits))
            j = int(rng.integers(0, num_qubits))
            if k == 0:
                qc.h(i)
            elif k == 1 and i != j:
                qc.cx(i, j)
            elif k == 2:
                qc.rz(float(rng.random()*2*np.pi- np.pi), i)
            elif k == 3 and i != j:
                qc.cp(np.pi/4.0, i, j)
            elif k == 4 and i != j:
                qc.cz(i, j)
            else:
                qc.x(i)
        qcs.append(qc)
    return qcs

def make_grover_circuit (num_qubits: int) -> QuantumCircuit:
    # Get a benchmark circuit on algorithmic level representing the GHZ state with 5 qubits
    qc_algorithmic_level = get_benchmark(
        benchmark="grover", level=BenchmarkLevel.ALG, circuit_size=num_qubits
    )

    print(qc_algorithmic_level.draw())
    
    return qc_algorithmic_level

def save_to_qasm(circuit: QuantumCircuit, file_path: str) -> None:
    path = Path(file_path)
    
    if not (path.suffix == ".qasm" and path.stem.endswith("_qasm2")):
        qasm2_path = path.with_name(path.stem + "_qasm2").with_suffix(".qasm")
    else:
        qasm2_path = path

    if not (path.suffix == ".qasm" and path.stem.endswith("_qasm3")):
        qasm3_path = path.with_name(path.stem + "_qasm3").with_suffix(".qasm")
    else:
        qasm3_path = path

    qasm2_str = qasm2.dumps(circuit)
    qasm2_path.parent.mkdir(parents=True, exist_ok=True)
    qasm2_path.write_text(qasm2_str)

    qasm3_str = qasm3.dumps(circuit)
    qasm3_path.parent.mkdir(parents=True, exist_ok=True)
    qasm3_path.write_text(qasm3_str)

if __name__ == "__main__":
    num_qubits = 11
    circuit = make_grover_circuit(num_qubits)
    print(circuit)
    save_to_qasm(circuit, f"../../src/test/Grover/grover_{num_qubits}_MQT.qasm")