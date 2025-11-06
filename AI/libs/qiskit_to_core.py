from typing import List, Tuple
from qiskit.circuit import QuantumCircuit
from .core_type import Core

def circuit_to_core_list(qc: QuantumCircuit) -> Tuple[int, List[Core]]:
    cores: List[Core] = []
    nq = qc.num_qubits
    q_to_i = {qb: i for i, qb in enumerate(qc.qubits)}

    def emit(tag: str, qubits: List[int], theta: float=0.0, phi: float=0.0, lam: float=0.0):
        c = Core(tag=tag, qubits=qubits, theta=theta, phi=phi, lam=lam)
        c.normalize()
        cores.append(c)

    for inst, qargs, _ in qc.data:
        name: str = inst.name.lower()
        qs = [q_to_i[q] for q in qargs]

        if name in ("x","y","z","h","s","sdg","t","tdg"):
            emit(name.upper(), qs)
        elif name in ("rz","rx","ry"):
            theta = float(inst.params[0]); emit(name.upper(), qs, theta=theta)
        elif name in ("p","u1"):
            theta = float(inst.params[0]); emit("P", qs, theta=theta)
        elif name == "u2":
            phi = float(inst.params[0]); lam = float(inst.params[1]); emit("U2", qs, phi=phi, lam=lam)
        elif name == "u3":
            theta = float(inst.params[0]); phi = float(inst.params[1]); lam = float(inst.params[2])
            emit("U3", qs, theta=theta, phi=phi, lam=lam)
        elif name in ("cx","cnot"):
            emit("CX", qs)
        elif name == "cz":
            emit("CZ", qs)
        elif name == "cp":
            phi = float(inst.params[0]); emit("CP", qs, theta=phi)
        elif name == "crz":
            theta = float(inst.params[0]); emit("CRZ", qs, theta=theta)
        elif name == "swap":
            emit("SWAP", qs)
        else:
            emit(name.upper(), qs)

    return nq, cores