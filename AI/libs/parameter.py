from dataclasses import dataclass
from pathlib import Path
import configparser, os, json
from typing import Any

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
with open(PROJECT_ROOT / "paths.json") as f:
    PATHS = json.load(f)

def get_path(key: str) -> Path:
    return PROJECT_ROOT / PATHS[key]

SETTING_PATH = get_path("setting_ini")

def _get(cfg: configparser.ConfigParser, section: str, key: str, cast: Any, default: Any):
    if cast is bool:
        return cfg.getboolean(section, key, fallback=default)
    if cast is int:
        return cfg.getint(section, key, fallback=default)
    if cast is float:
        return cfg.getfloat(section, key, fallback=default)
    return cfg.get(section, key, fallback=default)

@dataclass
class SchedulerHeuristics:
    alive: bool
    cost_diag: float
    cost_anti: float
    cost_perm: float
    cost_general: float

@dataclass
class SchedulerAI:
    alive: bool

@dataclass
class FuserHeuristics:
    alive: bool
    score_diag_per_gate: float
    score_same_axis_per_gate: float
    score_phase_gadget: float
    score_hcxh: float
    model_bonus: float

@dataclass
class FuserAI:
    alive: bool

@dataclass
class General:
    rl: bool
    window_size: int
    top_k_levels: int
    sig_dim: int
    gate_feat_dim: int
    max_qubits: int
    gamma: float
    lam: float
    lr: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    update_epochs: int
    device: str
    use_cpp_reward: bool
    cpp_reward_prob: float
    cpp_reward_alpha: float

@dataclass
class Parameter:
    scheduler_heuristics: SchedulerHeuristics
    scheduler_ai: SchedulerAI
    fuser_heuristics: FuserHeuristics
    fuser_ai: FuserAI
    general: General

    @staticmethod
    def load(ini_path: Path = INI_PATH) -> "Parameter":
        cfg = configparser.ConfigParser()
        cfg.read(ini_path)

        sh = SchedulerHeuristics(
            alive=_get(cfg, "Scheduler.Heuristics", "alive", bool, True),
            cost_diag=_get(cfg, "Scheduler.Heuristics", "cost_diag", float, 2.0),
            cost_anti=_get(cfg, "Scheduler.Heuristics", "cost_anti", float, 2.5),
            cost_perm=_get(cfg, "Scheduler.Heuristics", "cost_perm", float, 3.0),
            cost_general=_get(cfg, "Scheduler.Heuristics", "cost_general", float, 4.0),
        )
        sai = SchedulerAI(
            alive=_get(cfg, "Scheduler.AI", "alive", bool, True),
        )
        fh = FuserHeuristics(
            alive=_get(cfg, "Fuser.Heuristics", "alive", bool, True),
            score_diag_per_gate=_get(cfg, "Fuser.Heuristics", "score_diag_per_gate", float, 2.0),
            score_same_axis_per_gate=_get(cfg, "Fuser.Heuristics", "score_same_axis_per_gate", float, 1.0),
            score_phase_gadget=_get(cfg, "Fuser.Heuristics", "score_phase_gadget", float, 4.0),
            score_hcxh=_get(cfg, "Fuser.Heuristics", "score_hcxh", float, 3.0),
            model_bonus=_get(cfg, "Fuser.Heuristics", "model_bonus", float, 0.5),
        )
        fai = FuserAI(
            alive=_get(cfg, "Fuser.AI", "alive", bool, True),
        )
        gen = General(
            rl=_get(cfg, "General", "rl", bool, False),
            window_size=_get(cfg, "General", "window_size", int, 32),
            top_k_levels=_get(cfg, "General", "top_k_levels", int, 8),
            sig_dim=_get(cfg, "General", "sig_dim", int, 6),
            gate_feat_dim=_get(cfg, "General", "gate_feat_dim", int, 128),
            max_qubits=_get(cfg, "General", "max_qubits", int, 64),
            gamma=_get(cfg, "General", "gamma", float, 0.995),
            lam=_get(cfg, "General", "lam", float, 0.95),
            lr=_get(cfg, "General", "lr", float, 3e-4),
            clip_eps=_get(cfg, "General", "clip_eps", float, 0.2),
            ent_coef=_get(cfg, "General", "ent_coef", float, 0.01),
            vf_coef=_get(cfg, "General", "vf_coef", float, 0.5),
            update_epochs=_get(cfg, "General", "update_epochs", int, 5),
            device=_get(cfg, "General", "device", str, "cpu"),
            use_cpp_reward=_get(cfg, "General", "use_cpp_reward", bool, True),
            cpp_reward_prob=_get(cfg, "General", "cpp_reward_prob", float, 0.02),
            cpp_reward_alpha=_get(cfg, "General", "cpp_reward_alpha", float, 0.2),
        )
        return Parameter(
            scheduler_heuristics=sh,
            scheduler_ai=sai,
            fuser_heuristics=fh,
            fuser_ai=fai,
            general=gen,
        )