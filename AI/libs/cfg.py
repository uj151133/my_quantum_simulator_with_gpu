from dataclasses import dataclass

@dataclass
class Config:
    # Window / Signature
    window_size: int = 32
    top_k_levels: int = 8
    sig_dim: int = 6       # 4カテゴリのone-hot + 追加フラグ2つ（将来用）
    gate_feat_dim: int = 16
    max_qubits: int = 64

    # Cost weights（上位ほど重く、DIAG/ANTIは2、PERMは3、GENERALは4）
    w_top_decay: float = 0.7
    cost_diag: float = 2.0
    cost_anti: float = 2.0
    cost_perm: float = 3.0
    cost_general: float = 4.0
    lambda_node: float = 0.1

    # RL (PPO)
    gamma: float = 0.995
    lam: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    minibatch_size: int = 1024
    update_epochs: int = 5

    # PPO training config
    total_steps: int = 300_000
    rollout_steps: int = 4096

    # Device
    device: str = "cpu"   # 初期はCPU。CUDA環境なら "cuda" に変更