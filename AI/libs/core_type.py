from dataclasses import dataclass
from typing import List, Tuple

def _upper(s: str) -> str:
    return s.upper()

def _is_diag_tag(tag_u: str) -> bool:
    return tag_u in {"RZ","U1","P","S","T","Z","CZ","CP","CRZ","RZZ"}

def _is_symmetric_2q_tag(tag_u: str) -> bool:
    # 向きに不変な2Q（CZ/RZZ/SWAP など）
    return tag_u in {"CZ","RZZ","SWAP"}

def _infer_shape(tag_u: str, qubits: List[int]) -> str:
    if _is_diag_tag(tag_u):
        return "DIAG"
    if tag_u in {"X","SWAP"}:
        return "PERM"
    if tag_u in {"Y"}:
        return "ANTI"
    if tag_u == "FUSED":
        return "FUSED"
    return "GENERAL"

def _dedup_keep_order(xs: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for q in xs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

@dataclass
class Core:
    tag: str
    qubits: List[int]
    theta: float = 0.0
    phi: float = 0.0
    lam: float = 0.0
    shape: str = "GENERAL"
    handle: int = 0
    edge_nodes: int = 0

    # 2量子ゲート比較用
    def ordered_pair(self) -> Tuple[int,int]:
        return (self.qubits[0], self.qubits[1]) if len(self.qubits) == 2 else (-1, -1)

    def unordered_key(self) -> Tuple[int,int]:
        if len(self.qubits) != 2:
            return (-1, -1)
        a, b = self.qubits
        return (a, b) if a < b else (b, a)

    def is_symmetric_2q(self) -> bool:
        return _is_symmetric_2q_tag(_upper(self.tag))

    def normalize(self) -> None:
        self.tag = _upper(self.tag)
        # 並びは保持（control→target）。重複のみ除去。
        self.qubits = _dedup_keep_order(self.qubits)
        self.shape = _infer_shape(self.tag, self.qubits)