import random
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from AI.libs.ready_rules import build_ready_dag
from AI.libs.qiskit_to_core import circuit_to_core_list
from AI.libs.parameter import Parameter
from AI.libs.signature import build_signature
from AI.libs.bridge import Bridge


class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_ff=256, dropout=0.1, n_layers=2):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)

    def forward(self, x):
        return self.enc(x)


class SchedulerModel(nn.Module):
    """
    入れ替えポリシーモデル（完全版）
    - 入力: [gate_feature(params.general.gate_feat_dim) | signature(params.general.sig_dim*params.general.top_k_levels)]
    - 出力: スカラー・スコア
    - use_transformer=True で Transformer 前段に切替可
    """
    def __init__(self, input_dim: int, hidden: int = 128, use_transformer: bool = False, device: str = "cpu"):
        super().__init__()
        self.use_transformer = use_transformer
        self.device = torch.device(device)
        if use_transformer:
            self.pre = TransformerBlock(d_model=input_dim, nhead=4, dim_ff=hidden * 2, n_layers=2)
            self.head = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D] or [B, N, D]（Transformer時）
        return: [N] スコア
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


def _gate_feature(op: Dict[str, Any], dim: int, pos: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """
    ゲート単体の簡易特徴（タグ・最小量子ビット・shape cue・位置エンコーディング）
    """
    v = torch.zeros(dim, device=device)
    t = hash(op.get("tag", "")) % dim
    v[t] = 0.5
    qs = op.get("qubits", [])
    if qs:
        v[min(qs) % dim] += 0.5
    v[-1] = 1.0 if (op.get("shape", "GENERAL").upper() == "DIAG") else 0.0
    if pos is not None:
        v[(pos * 7) % dim] += 0.25
    return v


def _category_weight(op: Dict[str, Any], params: Parameter) -> float:
    shape = op.get("shape", "GENERAL").upper()
    if shape == "DIAG":
        return params.scheduler_heuristics.cost_diag
    if shape == "ANTI":
        return params.scheduler_heuristics.cost_anti
    if shape == "PERM":
        return params.scheduler_heuristics.cost_perm
    return params.scheduler_heuristics.cost_general


def _make_ready_features(ops: List[Dict[str, Any]], rlist: List[int], params: Parameter, device: str) -> torch.Tensor:
    """
    ready 候補それぞれについて [gate_feat | signature] を作る
    """
    sig = build_signature(ops, params).to(device)  # [sig_len]
    feats = []
    for i in rlist:
        gf = _gate_feature(ops[i], dim=params.general.gate_feat_dim, pos=i, device=device)
        feats.append(torch.cat([gf, sig], dim=0))  # [gate_feat_dim + sig_len]
    return torch.stack(feats, dim=0).to(device)     # [R, D]

_DIAG_TAGS = {"RZ","U1","P","S","T","Z","CZ","CP","CRZ","RZZ"}

def _as_dict(op):
    if isinstance(op, dict):
        return op
    tag = str(getattr(op, "tag", "")).upper()
    qubits = list(getattr(op, "qubits", []))
    d = {
        "tag": tag,
        "qubits": qubits,
        "theta": float(getattr(op, "theta", 0.0)),
        "phi": float(getattr(op, "phi", 0.0)),
        "lam": float(getattr(op, "lam", 0.0)),
    }
    shape = getattr(op, "shape", None)
    d["shape"] = str(shape) if isinstance(shape, str) and shape else ("DIAG" if tag in _DIAG_TAGS else "GENERAL")
    return d


def reorder_ops_by_model(
    ops: List[Dict[str, Any]],
    model,
    params: Parameter,
    device: str = "cpu",
    bridge: Optional[Bridge] = None,
    use_cpp_dag: bool = True,
) -> List[Dict[str, Any]]:
    ops = [_as_dict(o) for o in ops]
    N = len(ops)
    if N == 0:
        return []

    sig = build_signature(ops, params).to(device)   # [S]
    X = []
    for i in range(N):
        gf = _gate_feature(ops[i], dim=params.general.gate_feat_dim, pos=i, device=device)
        X.append(torch.cat([gf, sig], dim=0))
    X = torch.stack(X, dim=0)  # [N, D]
    with torch.no_grad():
        scores = model(X).detach().cpu().view(-1)

    # priority = score - weight
    prios = []
    for i in range(N):
        w = _category_weight(ops[i], params)  # 実装済み想定
        prios.append((float(scores[i].item()) - float(w), i))
    prios.sort(key=lambda x: -x[0])
    perm = [i for _, i in prios]

    if use_cpp_dag:
        if bridge is None:
            bridge = Bridge()
        order = bridge.legalize_order_by_dag(ops, perm)
    else:
        order = perm

    return [ops[i] for i in order]


def train_supervised_scheduler(
    model: SchedulerModel,
    train_qcs,
    params: Parameter,
    epochs: int = 5
) -> None:
    """
    教師あり学習（完全版）
    - データ: Qiskit回路 → Core(dict) 列
    - ラベル: ready集合の中から「shape=DIAG を優先、無ければ最小index」を正解とする疑似教師
    - 特徴: [gate_feat | signature]
    - 損失: CrossEntropy（ラベルスムージングあり）
    """
    model.train()
    opt = optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    device = str(model.device)

    for ep in range(epochs):
        total = 0.0
        for qc in train_qcs:
            nq, cores = circuit_to_core_list(qc)
            ops = [c.__dict__ for c in cores]

            preds, succs, ready = build_ready_dag(ops)
            rlist = sorted(list(ready))
            if not rlist:
                continue

            feats = _make_ready_features(ops, rlist, params, device)  # [R,D]

            # 疑似教師: shape=DIAG を優先、無ければ最小index
            diag_first = [i for i in rlist if ops[i].get("shape", "").upper() == "DIAG"]
            target_local = rlist.index(diag_first[0]) if diag_first else 0
            logits = model(feats)  # [R]

            loss = loss_fn(logits.unsqueeze(0), torch.tensor([target_local], device=model.device))
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())

        print(f"[SUP Scheduler] epoch={ep+1} loss={total:.4f}")


class _PPOValue(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, device: str = "cpu"):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.to(device)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_ppo_scheduler(
    model: SchedulerModel,
    bridge: Bridge,
    train_qcs,
    params: Parameter,
    episodes: int = 50
) -> None:
    """
    PPO（簡潔な完全版）
    - 各回路につき「最初のready選択」を1ステップのMDPとして扱う簡易近似
    - 行動: ready候補からサンプリング（softmax）
    - 報酬: shape=DIAG選択で+1、確率で C++ 実測改善（law適用差分）を加点（params に基づく混入）
    - 目的: Clip objective + 価値関数 + エントロピー正則化
    """
    model.train()
    device = str(model.device)

    input_dim = next(model.parameters()).shape[-1] if not model.use_transformer else None
    # 入力次元は ready特徴の D = gate_feat_dim + sig_len
    D = params.general.gate_feat_dim + params.general.sig_dim * params.general.top_k_levels
    vfn = _PPOValue(input_dim=D, hidden=128, device=device)
    opt = optim.Adam(list(model.parameters()) + list(vfn.parameters()), lr=params.general.lr)

    for ep in range(episodes):
        ep_return = 0.0

        for qc in train_qcs:
            nq, cores = circuit_to_core_list(qc)
            ops = [c.__dict__ for c in cores]

            preds, succs, ready = build_ready_dag(ops)
            rlist = sorted(list(ready))
            if not rlist:
                continue

            feats = _make_ready_features(ops, rlist, params, device)  # [R,D]
            logits = model(feats)                                  # [R]
            probs = torch.softmax(logits, dim=0)
            dist = torch.distributions.Categorical(probs=probs)
            a_local = dist.sample()
            logp_old = dist.log_prob(a_local).detach()
            act_idx = rlist[int(a_local.item())]
            feat_chosen = feats[a_local.item()].unsqueeze(0)       # [1,D]

            # 代理報酬: shape=DIAG なら +1
            reward = 1.0 if ops[act_idx].get("shape", "").upper() == "DIAG" else 0.0

            # 実測混入
            if params.general.use_cpp_reward and random.random() < params.general.cpp_reward_prob:
                try:
                    before = bridge.evaluate_runtime_for_ops(nq, ops, "ppo_before")["wall_time_ms"]
                    after_ops = bridge.optimize_ops(ops)  # law 適用効果
                    after = bridge.evaluate_runtime_for_ops(nq, after_ops, "ppo_after")["wall_time_ms"]
                    delta = (before - after) / max(1.0, before)
                    reward = (1.0 - params.general.cpp_reward_alpha) * reward + params.general.cpp_reward_alpha * max(0.0, float(delta))
                except Exception:
                    pass

            # Advantage（1ステップ近似）
            with torch.no_grad():
                v_pred = vfn(feat_chosen)          # [1]
            ret = torch.tensor([reward], device=device)
            adv = ret - v_pred                      # [1]

            # PPO 更新（複数回）
            for _ in range(params.general.update_epochs):
                logits_new = model(feats)                # [R]
                dist_new = torch.distributions.Categorical(probs=torch.softmax(logits_new, dim=0))
                logp = dist_new.log_prob(a_local)        # []
                ratio = torch.exp(logp - logp_old)       # []

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - params.general.clip_eps, 1.0 + params.general.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v_new = vfn(feat_chosen)
                v_loss = ((v_new - ret) ** 2).mean() * params.general.vf_coef
                ent = dist_new.entropy().mean() * params.general.ent_coef

                loss = policy_loss + v_loss - ent
                opt.zero_grad()
                loss.backward()
                opt.step()

            ep_return += float(reward)

        print(f"[PPO Scheduler] ep={ep+1} return={ep_return:.3f}")


def make_scheduler_chooser(model: SchedulerModel, params: Parameter, device: str = "cpu"):
    """
    推論用ユーティリティ: ops -> reordered_ops を返す関数を作る
    """
    def chooser(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return reorder_ops_by_model(ops, model, params=params, device=device)
    return chooser