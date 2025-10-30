from qiskit.circuit import QuantumCircuit
from libs.bridge.qiskit_to_ops import circuit_to_ops
from libs.bridge.cpp_client import CppClient
import time, statistics as stats

def gen_ring(n=6, depth=40):
    qc = QuantumCircuit(n)
    for i in range(depth):
        qc.h(i % n)
        qc.cx(i % n, (i+1) % n)
        qc.rz(0.7, (i+2) % n)
    return qc

def bench(n=6, depth=40, trials=30):
    qc = gen_ring(n, depth)
    nq, ops = circuit_to_ops(qc)
    client = CppClient(nq)

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = client.evaluate_chunk_runtime("bench", ops)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    print(f"n={n}, depth={depth}, trials={trials} -> ms: mean={stats.mean(times):.2f}, p50={stats.median(times):.2f}")

if __name__ == "__main__":
    bench()