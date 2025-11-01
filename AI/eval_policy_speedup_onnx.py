import argparse, time, statistics as stats
import numpy as np
import onnxruntime as ort
import torch
from qiskit.circuit import QuantumCircuit
from libs.cfg import Config
from libs.bridge.qiskit_to_ops import circuit_to_ops
from libs.bridge.cpp_client import CppClient

GATE_LIST = ["H","X","Y","Z","S","SDG","T","TDG","SX","RX","RY","RZ","P","U1","U2","U3","CX","CY","CZ","CP","CH","CRX","CRY","CRZ","CU","SWAP"]

def gate_one_hot(name: str, F: int):
    v = np.zeros(F, dtype=np.float32)
    try:
        i = GATE_LIST.index(name.upper())
        if i < F: v[i] = 1.0
    except ValueError:
        pass
    return v

def build_ready_dag(ops):
    last_on_qubit = {}
    preds = [set() for _ in range(len(ops))]
    succs = [set() for _ in range(len(ops))]
    for i, op in enumerate(ops):
        for q in op.get("qubits", []):
            if q in last_on_qubit:
                j = last_on_qubit[q]
                preds[i].add(j)
                succs[j].add(i)
            last_on_qubit[q] = i
    ready = {i for i in range(len(ops)) if not preds[i]}
    return preds, succs, ready

def schedule_with_onnx(ops, sess: ort.InferenceSession, cfg: Config):
    W, K, sd, F = cfg.window_size, cfg.top_k_levels, cfg.sig_dim, cfg.gate_feat_dim
    preds, succs, ready = build_ready_dag(ops)
    scheduled, done = [], set()
    sig = np.zeros((1, K, sd), dtype=np.float32)

    while len(scheduled) < len(ops):
        rlist = sorted(list(ready - done))
        if not rlist:
            left = [i for i in range(len(ops)) if i not in done]
            rlist = [left[0]]
        wfeat = np.zeros((1, W, F), dtype=np.float32)
        mask  = np.zeros((1, W), dtype=np.float32)
        for i, idx in enumerate(rlist[:W]):
            wfeat[0, i] = gate_one_hot(ops[idx]["gate_type"], F)
            mask[0, i] = 1.0

        logits, _ = sess.run(["logits","value"], {"sig":sig, "win":wfeat, "mask":mask})
        logits = logits + np.log(mask + 1e-8)
        a = int(np.argmax(logits, axis=-1)[0])
        pick_pos = a if a < len(rlist) else 0
        pick_idx = rlist[pick_pos]
        scheduled.append(pick_idx); done.add(pick_idx)
        for nxt in list(succs[pick_idx]):
            preds[nxt].discard(pick_idx)
            if not preds[nxt]: ready.add(nxt)
    return [ops[i] for i in scheduled]

def gen_ring(n=6, depth=40):
    qc = QuantumCircuit(n)
    for i in range(depth):
        qc.h(i % n); qc.cx(i % n, (i+1) % n); qc.rz(0.7, (i+2) % n)
    return qc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--depth", type=int, default=40)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    cfg = Config()
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    qc = gen_ring(args.n, args.depth)
    nq, ops = circuit_to_ops(qc)
    client = CppClient(nq)

    base_times, pol_times = [], []
    for _ in range(args.trials):
        t0 = time.perf_counter(); client.evaluate_chunk_runtime("base", ops); t1 = time.perf_counter()
        base_times.append((t1-t0)*1000)
    sch_ops = schedule_with_onnx(ops, sess, cfg)
    for _ in range(args.trials):
        t0 = time.perf_counter(); client.evaluate_chunk_runtime("pol", sch_ops); t1 = time.perf_counter()
        pol_times.append((t1-t0)*1000)

    base_mean, pol_mean = float(np.mean(base_times)), float(np.mean(pol_times))
    speedup = (base_mean - pol_mean)/base_mean*100.0
    print(f"baseline {base_mean:.2f} ms, policy {pol_mean:.2f} ms, speedup {speedup:.1f}%")

if __name__ == "__main__":
    main()