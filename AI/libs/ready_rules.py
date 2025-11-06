from typing import List, Dict, Any, Tuple, Set

def _is_diag_shape(op: Dict[str,Any]) -> bool:
    shp = op.get("shape", "").upper()
    if shp:
        return shp == "DIAG"
    # shape 未設定の古い dict では tag から推定（互換）
    t = op.get("tag","").upper()
    qs = op.get("qubits",[])
    if len(qs)==1: return t in {"RZ","U1","P","S","T","Z"}
    if len(qs)==2: return t in {"CZ","CP","CRZ","RZZ"}
    return False

def build_ready_dag(ops: List[Dict[str,Any]]) -> Tuple[List[Set[int]], List[Set[int]], Set[int]]:
    """
    安全な可換のみ許容:
      - DIAG×DIAG は共有ビットでも依存なし
      - 同一control・別target の CX 同士は依存なし
      - それ以外は「同じ量子ビットに触れた直前」に依存
    """
    N = len(ops)
    preds: List[Set[int]] = [set() for _ in range(N)]
    succs: List[Set[int]] = [set() for _ in range(N)]
    last_touch: Dict[int,int] = {}
    for i, op in enumerate(ops):
        for q in op.get("qubits", []):
            if q in last_touch:
                j = last_touch[q]
                a, b = ops[j], op
                both_diag = _is_diag_shape(a) and _is_diag_shape(b)
                cx_same_ctrl_diff_tgt = (
                    a.get("tag","").upper() in {"CX","CNOT"} and
                    b.get("tag","").upper() in {"CX","CNOT"} and
                    len(a.get("qubits",[]))==2 and len(b.get("qubits",[]))==2 and
                    a["qubits"][0]==b["qubits"][0] and a["qubits"][1]!=b["qubits"][1]
                )
                if not (both_diag or cx_same_ctrl_diff_tgt):
                    preds[i].add(j); succs[j].add(i)
            last_touch[q] = i
    ready = {i for i in range(N) if not preds[i]}
    return preds, succs, ready