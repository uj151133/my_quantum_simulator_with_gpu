from qiskit.circuit import QuantumCircuit
import qmdd_core as m
from libs.bridge.qiskit_to_ops import circuit_to_ops
from libs.bridge.cpp_client import CppClient

def main():
    qc = QuantumCircuit(4)
    qc.h(0); qc.cx(0, 2); qc.rz(0.7, 2); qc.sx(1)

    nq, ops = circuit_to_ops(qc)
    client = CppClient(nq)
    res = client.evaluate_chunk_runtime("smoke", ops)
    print("metrics:", res)

if __name__ == "__main__":
    main()