from dataclasses import dataclass

@dataclass
class Config:
    # Window / Signature
    window_size: int = 32
    top_k_levels: int = 8        # 上位タグK
    sig_dim: int = 6             # shapeヒスト+タグハッシュの1レベル当たりの次元
    gate_feat_dim: int = 128     # 1ゲートの特徴次元（モデル入力の基底部分）
    max_qubits: int = 64

    # Scheduler bias
    cost_diag: float = 2.0
    cost_anti: float = 2.5
    cost_perm: float = 3.0
    cost_general: float = 4.0

    # Fusion rule scores
    fusion_score_diag_per_gate: float = 2.0
    fusion_score_same_axis_per_gate: float = 1.0
    fusion_score_phase_gadget: float = 4.0
    fusion_score_hcxh: float = 3.0
    fusion_model_bonus: float = 0.5

    # RL (PPO)
    gamma: float = 0.995
    lam: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    update_epochs: int = 5

    # Device
    device: str = "cpu"

    use_cpp_reward: bool = True
    cpp_reward_prob: float = 0.02
    cpp_reward_alpha: float = 0.2