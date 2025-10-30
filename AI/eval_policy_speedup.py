import argparse, time, statistics as stats
import numpy as np
import torch
from qiskit.circuit import QuantumCircuit
from libs.cfg import Config
from libs.models.pointer_policy import PointerPolicy
from libs.io.checkpoint import load_checkpoint
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
    # 各ゲートに対して「同じ量子ビットに触る直前のゲート」を依存先とするDAGを構成
    last_on_qubit = {}
    preds = [set() for _ in range(len(ops))]
    succs = [set() for _ in range(len(ops))]
    for i, op in enumerate(ops):
        qset = tuple(sorted(op.get("qubits", [])))
        for q in qset:
            if q in last_on_qubit:
                j = last_on_qubit[q]
                preds[i].add(j)
                succs[j].add(i)
            last_on_qubit[q] = i
    ready = {i for i in range(len(ops)) if not preds[i]}
    return preds, succs, ready

def schedule_with_policy(ops, model: PointerPolicy, cfg: Config, device: str):
    W, K, sd, F = cfg.window_size, cfg.top_k_levels, cfg.sig_dim, cfg.gate_feat_dim
    preds, succs, ready = build_ready_dag(ops)
    scheduled = []
    done = set()
    # 簡易sig（ゼロ）。必要ならコスト特徴に発展可
    sig = torch.zeros(1, K, sd, dtype=torch.float32, device=device)
    model.eval()

    while len(scheduled) < len(ops):
        # レディ集合からウィンドウを組む
        rlist = sorted(list(ready - done))
        if not rlist:
            # 依存解決のための保険（循環はない想定）
            left = [i for i in range(len(ops)) if i not in done]
            rlist = [left[0]]
        # ウィンドウ埋めとmask作成
        wfeat = np.zeros((W, F), dtype=np.float32)
        mask = np.zeros((W,), dtype=np.float32)
        for i, idx in enumerate(rlist[:W]):
            wfeat[i] = gate_one_hot(ops[idx]["gate_type"], F)
            mask[i] = 1.0
        win = torch.tensor(wfeat[None, ...], dtype=torch.float32, device=device)
        msk = torch.tensor(mask[None, ...], dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, _ = model({"sig": sig, "win": win, "mask": msk})
            # マスクはモデル内でも加算される前提だが、念のため無効位置を極小へ
            logits = logits + torch.log(msk + 1e-8)
            a = torch.argmax(logits, dim=-1).item()
        # 選択
        pick_pos = int(a) if a < len(rlist) else 0
        pick_idx = rlist[pick_pos]
        scheduled.append(pick_idx)
        done.add(pick_idx)
        # 依存更新
        for nxt in list(succs[pick_idx]):
            preds[nxt].discard(pick_idx)
            if not preds[nxt]: ready.add(nxt)
    return [ops[i] for i in scheduled]

def gen_ring(n=6, depth=40):
    qc = QuantumCircuit(n)
    for i in range(depth):
        qc.h(i % n); qc.cx(i % n, (i+1) % n); qc.rz(0.7, (i+2) % n)
    return qc

def run_once(model, cfg, device, n, depth, trials=10):
    qc = gen_ring(n, depth)
    nq, ops = circuit_to_ops(qc)
    client = CppClient(nq)

    # baseline
    base_times = []
    for _ in range(trials):
        t0 = time.perf_counter(); client.evaluate_chunk_runtime("base", ops); t1 = time.perf_counter()
        base_times.append((t1-t0)*1000)

    # policy
    sch_ops = schedule_with_policy(ops, model, cfg, device)
    pol_times = []
    for _ in range(trials):
        t0 = time.perf_counter(); client.evaluate_chunk_runtime("pol", sch_ops); t1 = time.perf_counter()
        pol_times.append((t1-t0)*1000)

    base_mean = stats.mean(base_times); pol_mean = stats.mean(pol_times)
    speedup = (base_mean - pol_mean) / base_mean * 100.0
    print(f"n={n}, depth={depth}, trials={trials} -> baseline {base_mean:.2f} ms, policy {pol_mean:.2f} ms, speedup {speedup:.1f}%")
    return base_mean, pol_mean, speedup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--depth", type=int, default=40)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    cfg = Config()
    cfg.device = args.device
    model = PointerPolicy(cfg).to(cfg.device)
    load_checkpoint(model, args.ckpt, map_location=cfg.device)
    run_once(model, cfg, cfg.device, args.n, args.depth, args.trials)

if __name__ == "__main__":
    main()