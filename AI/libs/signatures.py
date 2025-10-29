import enum
import numpy as np
from dataclasses import dataclass
from typing import List

class Block(enum.IntEnum):
    DIAG = 0
    ANTI = 1
    PERM = 2
    GENERAL = 3

@dataclass
class LevelSig:
    block: Block
    f1: int = 0
    f2: int = 0

@dataclass
class Signature:
    levels: List[LevelSig]  # top-first

def encode_level(ls: LevelSig, sig_dim: int) -> np.ndarray:
    v = np.zeros(sig_dim, dtype=np.float32)
    v[int(ls.block)] = 1.0
    v[-2] = float(ls.f1)
    v[-1] = float(ls.f2)
    return v

def encode_signature(sig: Signature, K: int, sig_dim: int) -> np.ndarray:
    out = np.zeros((K, sig_dim), dtype=np.float32)
    for i in range(min(K, len(sig.levels))):
        out[i] = encode_level(sig.levels[i], sig_dim)
    return out

def branch_cost(block: Block, cfg) -> float:
    if block == Block.DIAG: return cfg.cost_diag
    if block == Block.ANTI: return cfg.cost_anti
    if block == Block.PERM: return cfg.cost_perm
    return cfg.cost_general

def signature_cost(sig: Signature, cfg) -> float:
    c = 0.0
    w = 1.0
    for ls in sig.levels[:cfg.top_k_levels]:
        c += w * branch_cost(ls.block, cfg)
        w *= cfg.w_top_decay
    return c

def update_rule_block(a: Block, b: Block) -> Block:
    if a == Block.DIAG and b == Block.DIAG: return Block.DIAG
    if a == Block.DIAG and b == Block.ANTI: return Block.ANTI
    if a == Block.ANTI and b == Block.DIAG: return Block.ANTI
    if a == Block.PERM and b == Block.DIAG: return Block.PERM
    return Block.GENERAL

def simulate_update(sig: Signature, gate_sig: Signature, K: int) -> Signature:
    out = []
    for i in range(K):
        a = sig.levels[i] if i < len(sig.levels) else LevelSig(Block.DIAG,0,0)
        b = gate_sig.levels[i] if i < len(gate_sig.levels) else LevelSig(Block.DIAG,0,0)
        out.append(LevelSig(update_rule_block(a.block, b.block), a.f1|b.f1, a.f2|b.f2))
    return Signature(out)

# スタブ変換：ゲート→シグネチャ（上位Kの作用レベルに base を立てる）
def gate_to_signature_stub(gate_type:str, acting_levels:List[int], K:int) -> Signature:
    if gate_type in ["Z","Rz","CZ","CP"]: base = Block.DIAG
    elif gate_type in ["X","H","Rx","Ry","SWAP","CX"]: base = Block.ANTI if gate_type!="CX" else Block.GENERAL
    else: base = Block.GENERAL
    levels = [LevelSig(Block.DIAG,0,0) for _ in range(K)]
    for i in acting_levels:
        if 0 <= i < K: levels[i] = LevelSig(base,0,0)
    return Signature(levels)