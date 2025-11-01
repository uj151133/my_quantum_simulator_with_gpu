import gymnasium as gym
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from ..cfg import Config
from ..signatures import Signature, encode_signature, gate_to_signature_stub, simulate_update, signature_cost
from ..cost_estimator import HeuristicCostEstimator
from ..bridge.cpp_client import CppClient

@dataclass
class GateItem:
    gate_type: str
    acting_levels: List[int]
    is_diag: int
    is_perm: int

class QMDDSchedulingEnv(gym.Env):
    """
    ゲート並べ替え（スケジューリング）専用の環境。
    - 1手でウィンドウから1ゲートを選ぶ
    - 報酬は署名コストの減少（負の増分）
    - エピソード終端で低頻度にC++実測ボーナスをブレンド（任意）
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, cpp: CppClient = None):
        super().__init__()
        self.cfg = cfg
        self.cost = HeuristicCostEstimator(cfg)

        # C++クライアントは nq ごとにキャッシュして使い回し
        self._client_cache: Dict[int, CppClient] = {}
        if cpp is not None:
            self._client_cache[cfg.max_qubits] = cpp

        W, K, sd, fd = cfg.window_size, cfg.top_k_levels, cfg.sig_dim, cfg.gate_feat_dim
        self.observation_space = gym.spaces.Dict({
            "sig": gym.spaces.Box(low=0, high=1, shape=(K, sd), dtype=np.float32),
            "win": gym.spaces.Box(low=-1, high=1, shape=(W, fd), dtype=np.float32),
            "mask": gym.spaces.Box(low=0, high=1, shape=(W,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(W)

        self.action_history: List[int] = []
        self.reset()

    # ---------- utils ----------
    def _get_client(self, nq: int) -> CppClient:
        cli = self._client_cache.get(nq)
        if cli is None:
            cli = CppClient(nq)
            self._client_cache[nq] = cli
        return cli

    def _make_dummy_window(self) -> List[GateItem]:
        """
        デモ用のダミー・ウィンドウ（本番は実回路ベースに差し替え）
        """
        types = ["Rz","Rx","H","CZ","CX","CP","Z","X"]
        W, K = self.cfg.window_size, self.cfg.top_k_levels
        win: List[GateItem] = []
        for _ in range(W):
            gt = np.random.choice(types, p=[0.18,0.20,0.10,0.16,0.18,0.10,0.04,0.04])
            is_diag = 1 if gt in ["Rz","Z","CZ","CP"] else 0
            is_perm = 1 if gt in ["X"] else 0
            act = sorted(set(np.random.randint(0, K, size=np.random.randint(1,3)).tolist()))
            win.append(GateItem(gt, act, is_diag, is_perm))
        return win

    # ---------- gym api ----------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
        self.action_history = []
        self.prefix_sig = Signature([])   # 既定: 全てDIAG扱いから開始
        self.window = self._make_dummy_window()
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        cfg = self.cfg
        sig_enc = encode_signature(self.prefix_sig, cfg.top_k_levels, cfg.sig_dim)
        win_feat = np.zeros((cfg.window_size, cfg.gate_feat_dim), dtype=np.float32)
        mask = np.zeros((cfg.window_size,), dtype=np.float32)

        for i, g in enumerate(self.window):
            if g is None:
                continue
            feat = np.zeros((cfg.gate_feat_dim,), dtype=np.float32)
            # 位置のライトな符号化（簡易）。本番はゲートone-hot等に差し替え可
            feat[min(i, cfg.gate_feat_dim-3)] = 1.0
            # 作用レベルの粗い要約（最小レベルを0..1に正規化）
            feat[-3] = float(min(g.acting_levels)) / max(1, cfg.top_k_levels-1) if g.acting_levels else 0.0
            feat[-2] = float(g.is_diag)
            feat[-1] = float(g.is_perm)
            win_feat[i] = feat
            mask[i] = 1.0

        return {"sig": sig_enc, "win": win_feat, "mask": mask}

    def step(self, action: int):
        cfg = self.cfg
        if not (0 <= action < cfg.window_size) or self.window[action] is None:
            return self._obs(), -0.1, False, False, {}

        self.action_history.append(int(action))

        g = self.window[action]
        gate_sig = gate_to_signature_stub(g.gate_type, g.acting_levels, cfg.top_k_levels)

        before = signature_cost(self.prefix_sig, cfg)
        self.prefix_sig = simulate_update(self.prefix_sig, gate_sig, cfg.top_k_levels)
        after = signature_cost(self.prefix_sig, cfg)
        reward = -float(after - before)

        # 消化
        self.window[action] = None
        self.t += 1
        done = self.t >= cfg.window_size or all(x is None for x in self.window)

        # 終端でCPP実測ボーナス（低確率）
        if done and cfg.use_cpp_reward and random.random() < cfg.cpp_reward_prob and len(self.action_history) > 0:
            try:
                cpp_bonus = self._cpp_reward_bonus(self.action_history)
                reward = (1.0 - cfg.cpp_reward_alpha) * reward + cfg.cpp_reward_alpha * cpp_bonus
            except Exception:
                pass

        return self._obs(), reward, done, False, {}

    # ---------- C++ 実測混入 ----------
    def _cpp_reward_bonus(self, history: List[int]) -> float:
        """
        小回路で baseline vs policy 順の C++ 実測を比較し、[-0.5,+0.5] ボーナスを返す
        """
        from qiskit.circuit import QuantumCircuit
        from ..bridge.qiskit_to_ops import circuit_to_ops

        def gen_ring(n=6, depth=40):
            qc = QuantumCircuit(n)
            for i in range(depth):
                qc.h(i % n); qc.cx(i % n, (i+1) % n); qc.rz(0.7, (i+2) % n)
            return qc

        n = min(self.cfg.max_qubits, 6)
        qc = gen_ring(n, depth=40)
        nq, ops = circuit_to_ops(qc)
        client = self._get_client(nq)

        def run_ms(o):
            res = client.evaluate_chunk_runtime("eval", o)
            return float(res["wall_time_ms"])

        trials = 3
        base = np.mean([run_ms(ops) for _ in range(trials)])
        sch_ops = self._schedule_by_history(ops, history, self.cfg.window_size)
        pol = np.mean([run_ms(sch_ops) for _ in range(trials)])
        if base <= 0:
            return 0.0
        speedup = (base - pol) / base
        return float(np.clip(speedup, -0.5, 0.5))

    # ---------- ready DAG（安全な可換則込み） ----------
    @staticmethod
    def _is_diag_op(op: Dict[str, Any]) -> bool:
        t = op.get("gate_type","").upper()
        qs = op.get("qubits", [])
        if len(qs) == 1:
            return t in {"RZ","U1","P","S","T","Z"}
        if len(qs) == 2:
            return t in {"CZ","CP","CRZ","RZZ"}
        return False

    def _build_ready_dag(self, ops: List[Dict[str, Any]]) -> Tuple[List[set], List[set], set]:
        """
        安全な可換則:
          - 対角×対角は依存を張らない
          - 同一control・別target の CX 同士は依存を張らない
        それ以外は「同じ量子ビットに触れた直前ゲート」に依存を張る
        """
        N = len(ops)
        preds = [set() for _ in range(N)]
        succs = [set() for _ in range(N)]
        last_touch: Dict[int, int] = {}

        for i, op in enumerate(ops):
            qs = op.get("qubits", [])
            for q in qs:
                if q in last_touch:
                    j = last_touch[q]
                    a, b = ops[j], op
                    both_diag = self._is_diag_op(a) and self._is_diag_op(b)
                    cx_same_ctrl_diff_tgt = (
                        a.get("gate_type","").upper() in {"CX","CNOT"} and
                        b.get("gate_type","").upper() in {"CX","CNOT"} and
                        len(a.get("qubits",[]))==2 and len(b.get("qubits",[]))==2 and
                        a["qubits"][0] == b["qubits"][0] and a["qubits"][1] != b["qubits"][1]
                    )
                    # 安全可換でなければ依存を張る
                    if not (both_diag or cx_same_ctrl_diff_tgt):
                        preds[i].add(j)
                        succs[j].add(i)
                last_touch[q] = i

        ready = {i for i in range(N) if not preds[i]}
        return preds, succs, ready

    def _schedule_by_history(self, ops: List[Dict[str, Any]], history: List[int], W: int) -> List[Dict[str, Any]]:
        preds, succs, ready = self._build_ready_dag(ops)
        scheduled, done = [], set()
        hlen = len(history) if history else 0
        t = 0
        while len(scheduled) < len(ops):
            rlist = sorted(list(ready - done))
            if not rlist:
                left = [i for i in range(len(ops)) if i not in done]
                rlist = [left[0]]
            idx_in_ready = history[t % hlen] if hlen > 0 else 0
            if idx_in_ready >= len(rlist):
                idx_in_ready = len(rlist) - 1
            pick_idx = rlist[idx_in_ready]
            scheduled.append(pick_idx); done.add(pick_idx)
            # update successors
            for nxt in list(succs[pick_idx]):
                preds[nxt].discard(pick_idx)
                if not preds[nxt]:
                    ready.add(nxt)
            t += 1
        return [ops[i] for i in scheduled]