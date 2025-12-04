from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from AI.libs.fusion_rules import score_candidates
from AI.libs.qiskit_to_core import circuit_to_core_list
from AI.libs.parameter import Parameter
from AI.libs.bridge import Bridge
from AI.libs.signature import build_signature


class FuserModel(nn.Module):
    """
    融合ポリシーモデル（完全版）
    - 入力ベクトルは「ゲート特徴（params.general.gate_feat_dim）」と「シグネチャ（params.general.sig_dim*params.general.top_k_levels）」を連結
    - forward: 位置ごとのスカラー・スコアを返す
    - propose: descs（Core互換dict配列）から非重複の融合レンジ [(s,e)] を返す
      ・ルールスコア（params 駆動）とモデル出力のピークを合成し、スコア降順で非重複採択
    """
    def __init__(self, input_dim: int, hidden: int = 128, use_transformer: bool = False, device: str = "cpu"):
        super().__init__()
        self.use_transformer = use_transformer
        self.device = torch.device(device)
        if use_transformer:
            enc = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True)
            self.pre = nn.TransformerEncoder(enc, num_layers=2)
            self.head = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D] または [B, N, D]（transformer利用時）
        return: [N] のスコア（大きいほど優先度が高い）
        """
        x = x.to(self.device)
        if self.use_transformer:
            if x.dim() == 2:
                x = x.unsqueeze(0)  # [1,N,D]
            h = self.pre(x)         # [B,N,D]
            out = self.head(h)      # [B,N,1]
            return out.squeeze(-1).mean(dim=0)  # [N]
        else:
            return self.net(x).squeeze(-1)

    def _gate_feature(self, op: Dict[str, Any], dim: int, pos: int = 0) -> torch.Tensor:
        """
        ゲート単体の簡易特徴（タグ・最小量子ビット・shape cue・位置エンコーディング）
        """
        v = torch.zeros(dim, device=self.device)
        t = hash(op.get("tag", "")) % dim
        v[t] = 0.5
        qs = op.get("qubits", [])
        if qs:
            v[min(qs) % dim] += 0.5
        v[-1] = 1.0 if (op.get("shape", "GENERAL").upper() == "DIAG") else 0.0
        v[(pos * 7) % dim] += 0.25
        return v

    def _concat_feat_for_all(self, ops: List[Dict[str, Any]], params: Parameter) -> torch.Tensor:
        """
        全位置について [gate_feat | signature] を作って [N, D] で返す
        """
        sig = build_signature(ops, params).to(self.device)
        feats = []
        for idx, op in enumerate(ops):
            gf = self._gate_feature(op, dim=params.general.gate_feat_dim, pos=idx)
            feats.append(torch.cat([gf, sig], dim=0))
        return torch.stack(feats, dim=0)  # [N, gate_feat_dim + sig_len]

    def propose(
        self,
        descs: List[Dict[str, Any]],
        params: Optional[Parameter] = None,
        min_len: int = 2,
        peak_topk: int = 4
    ) -> List[Tuple[int, int]]:
        """
        モデル推論に基づく融合レンジの提案
        - ルール候補（score_candidates）を土台に、モデルのピーク位置から短レンジも追加
        - 優先度は ルールスコア と モデルボーナス（params.fuser_heuristics.model_bonus）で合成
        - 非重複化して [(s,e)] を返す
        """
        if params is None:
            params = Parameter.load()
        n = len(descs)
        if n == 0:
            return []

        # 1) ルールに基づく候補（スコア付き）
        scored = score_candidates(descs, params=params)  # [(s,e,rule_score)]

        # 2) モデルスコアのピークから短い候補を追加（対角中心を優遇）
        x = self._concat_feat_for_all(descs, params)  # [N, D]
        with torch.no_grad():
            pos_scores = self.forward(x)          # [N]
        k = min(peak_topk, n)
        for idx in torch.topk(pos_scores, k=k).indices.tolist():
            # shape=DIAG を中心に短い帯を形成
            if descs[idx].get("shape", "").upper() == "DIAG":
                s = max(0, idx - 1)
                e = min(n - 1, idx + 1)
                if (e - s + 1) >= min_len:
                    scored.append((s, e, params.fuser_heuristics.model_bonus))
            else:
                # 非対角は単独では risky。隣接が対角なら最小レンジ化
                s = idx
                e = min(n - 1, idx + 1)
                if s < e and (descs[e].get("shape", "").upper() == "DIAG"):
                    scored.append((s, e, params.fuser_heuristics.model_bonus * 0.5))

        # 3) 優先度順で非重複化
        scored.sort(key=lambda x: (-x[2], x[0], x[1]))
        taken: List[Tuple[int, int]] = []
        covered = [False] * n
        for s, e, _sc in scored:
            if s < 0 or e < 0 or s >= n or e >= n or s > e:
                continue
            if any(covered[i] for i in range(s, e + 1)):
                continue
            taken.append((s, e))
            for i in range(s, e + 1):
                covered[i] = True
        return taken


def train_supervised_fuser(model: FuserModel, train_qcs, params: Parameter, epochs: int = 5) -> None:
    """
    教師あり学習（完全版）
    - 正解は score_candidates の上位候補の中心位置（スコアに比例した重み）
    - モデルは位置スコアを出す。回帰（MSE）で中心ピークを上げる
    """
    model.train()
    opt = optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        total = 0.0
        for qc in train_qcs:
            nq, cores = circuit_to_core_list(qc)
            ops = [c.__dict__ for c in cores]
            scored = score_candidates(ops, params=params)  # [(s,e,rule_sc)]
            if not scored:
                continue

            x = model._concat_feat_for_all(ops, params)  # [N,D]
            # 教師信号: 候補中心に 1.0、他 0（候補スコアに重み）
            y = torch.zeros(x.size(0), device=model.device)
            w = torch.zeros_like(y)

            for (s, e, sc) in scored:
                idx = (s + e) // 2
                y[idx] = 1.0
                w[idx] = max(1.0, float(sc))

            pred = model(x)  # [N]
            # 重み付きMSE
            loss = ((pred - y) ** 2 * (1.0 + w)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"[SUP Fuser] epoch={ep+1} loss={total:.4f}")


def train_ppo_fuser(
    model: FuserModel,
    bridge: Bridge,
    train_qcs,
    params: Parameter,
    episodes: int = 50,
    peak_topk: int = 4
) -> None:
    """
    PPO までの複雑さは不要という前提で、安定な方策勾配（REINFORCE風）を実装
    - 行動: モデルスコアのピークから短レンジ生成
    - 報酬: ルールスコア合計 + （確率で）C++実測改善の正規化加点
    - ルール候補との合成で非重複化してレンジ集合を作る
    """
    model.train()
    opt = optim.Adam(model.parameters(), lr=params.lr)

    for ep in range(episodes):
        ep_return = 0.0
        for qc in train_qcs:
            nq, cores = circuit_to_core_list(qc)
            ops = [c.__dict__ for c in cores]
            n = len(ops)
            if n == 0:
                continue

            # ルール候補
            base_scored = score_candidates(ops, params=params)  # [(s,e,rule)]
            base_ranges = {(s, e) for (s, e, _sc) in base_scored}

            # モデル候補（ピーク top-k）
            x = model._concat_feat_for_all(ops, params)      # [N,D]
            scores = model(x)                              # [N]
            topk = min(peak_topk, n)
            peak_idx = torch.topk(scores, k=topk).indices.tolist()

            # ピークから短レンジ生成（DIAG優先）
            model_scored: List[Tuple[int, int, float]] = []
            for idx in peak_idx:
                if ops[idx].get("shape", "").upper() == "DIAG":
                    s = max(0, idx - 1)
                    e = min(n - 1, idx + 1)
                    if e > s:
                        model_scored.append((s, e, params.fuser_heuristics.model_bonus))
                else:
                    s = idx
                    e = min(n - 1, idx + 1)
                    if s < e and ops[e].get("shape", "").upper() == "DIAG":
                        model_scored.append((s, e, params.fuser_heuristics.model_bonus * 0.5))

            # 合成して非重複化（優先度で）
            merged_scored = base_scored + model_scored
            merged_scored.sort(key=lambda x: (-x[2], x[0], x[1]))
            taken: List[Tuple[int, int]] = []
            covered = [False] * n
            for s, e, _sc in merged_scored:
                if any(covered[i] for i in range(s, e + 1)):
                    continue
                taken.append((s, e))
                for i in range(s, e + 1):
                    covered[i] = True

            # 報酬: ルールスコア合計（代理）+ 実測の改善（確率で）
            proxy = sum((e - s + 1) for (s, e) in taken)  # 簡易: 長いほど良い
            reward = float(proxy)

            if params.general.use_cpp_reward:
                import random
                if random.random() < params.general.cpp_reward_prob:
                    try:
                        before = bridge.evaluate_runtime_for_ops(nq, ops, "fuser_before")["wall_time_ms"]
                        # lawでIRを最適化したときの改善を加点（融合そのものはC++実行時に行われる）
                        after_ops = bridge.optimize_ops(ops)
                        after = bridge.evaluate_runtime_for_ops(nq, after_ops, "fuser_after")["wall_time_ms"]
                        delta = (before - after) / max(1.0, before)
                        reward += max(0.0, float(delta)) * params.general.cpp_reward_alpha
                    except Exception:
                        pass

            # REINFORCE風の更新（スコアの平均を上げる）
            loss = -scores.mean() * reward
            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_return += reward

        print(f"[PPO Fuser] ep={ep+1} return={ep_return:.3f}")