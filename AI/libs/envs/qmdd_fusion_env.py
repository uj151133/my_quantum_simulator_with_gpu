import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
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

class QMDDFusionEnv(gym.Env):
    """
    1手でウィンドウから1ゲートを選ぶ環境。
    報酬は署名コストの減少（負の増分）。C++実測を混ぜる拡張は bridge 側で可能。
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, cpp: CppClient = None):
        super().__init__()
        self.cfg = cfg
        self.cpp = cpp or CppClient()
        self.cost = HeuristicCostEstimator(cfg)

        W, K, sd, fd = cfg.window_size, cfg.top_k_levels, cfg.sig_dim, cfg.gate_feat_dim
        self.observation_space = gym.spaces.Dict({
            "sig": gym.spaces.Box(low=0, high=1, shape=(K, sd), dtype=np.float32),
            "win": gym.spaces.Box(low=-1, high=1, shape=(W, fd), dtype=np.float32),
            "mask": gym.spaces.Box(low=0, high=1, shape=(W,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(W)
        self.reset()

    def _make_dummy_window(self) -> List[GateItem]:
        types = ["Rz","Rx","H","CZ","CX","CP","Z","X"]
        W, K = self.cfg.window_size, self.cfg.top_k_levels
        win = []
        for _ in range(W):
            gt = np.random.choice(types, p=[0.18,0.2,0.1,0.16,0.18,0.1,0.04,0.04])
            is_diag = 1 if gt in ["Rz","Z","CZ","CP"] else 0
            is_perm = 1 if gt in ["X"] else 0
            act = sorted(set(np.random.randint(0, K, size=np.random.randint(1,3)).tolist()))
            win.append(GateItem(gt, act, is_diag, is_perm))
        return win

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.prefix_sig = Signature([])  # 既定は上位Kが全て DIAG として扱う
        self.window = self._make_dummy_window()
        self.t = 0
        self.chunk_ops: List[Dict[str,Any]] = []
        return self._obs(), {}

    def _obs(self):
        cfg = self.cfg
        sig_enc = encode_signature(self.prefix_sig, cfg.top_k_levels, cfg.sig_dim)
        win_feat = np.zeros((cfg.window_size, cfg.gate_feat_dim), dtype=np.float32)
        mask = np.zeros((cfg.window_size,), dtype=np.float32)
        for i, g in enumerate(self.window):
            if g is None: continue
            feat = np.zeros((cfg.gate_feat_dim,), dtype=np.float32)
            # 簡易エンコード（実運用ではタイプone-hotへ差し替え推奨）
            feat[min(i, cfg.gate_feat_dim-3)] = 1.0
            feat[-3] = float(min(g.acting_levels) if g.acting_levels else 0)/max(1,cfg.top_k_levels-1)
            feat[-2] = float(g.is_diag)
            feat[-1] = float(g.is_perm)
            win_feat[i] = feat
            mask[i] = 1.0
        return {"sig": sig_enc, "win": win_feat, "mask": mask}

    def step(self, action: int):
        cfg = self.cfg
        if not (0 <= action < cfg.window_size) or self.window[action] is None:
            return self._obs(), -0.1, False, False, {}

        g = self.window[action]
        gate_sig = gate_to_signature_stub(g.gate_type, g.acting_levels, cfg.top_k_levels)

        before = signature_cost(self.prefix_sig, cfg)
        self.prefix_sig = simulate_update(self.prefix_sig, gate_sig, cfg.top_k_levels)
        after = signature_cost(self.prefix_sig, cfg)
        delta = after - before
        reward = -float(delta)

        # 実測混入はここで self.cpp.evaluate_chunk_runtime(...) を呼んで reward にブレンド

        self.window[action] = None
        self.t += 1
        done = self.t >= cfg.window_size or all(x is None for x in self.window)
        return self._obs(), reward, done, False, {}