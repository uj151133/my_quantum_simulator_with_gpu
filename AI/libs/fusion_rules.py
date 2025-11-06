from typing import List, Tuple, Dict, Any, Optional
from AI.libs.config import Config

def score_candidates(descs: List[Dict[str, Any]], cfg: Optional[Config] = None) -> List[Tuple[int, int, float]]:
    """
    ルールに基づく優先度スコア（Config 駆動）を返す。
    FUSED shape は基本スキップ（既に融合済み）。
    """
    if cfg is None:
        cfg = Config()
    out: List[Tuple[int,int,float]] = []
    n = len(descs)
    if n == 0:
        return out

    def is_diag(i: int) -> bool:
        return descs[i].get("shape","").upper() == "DIAG"

    def is_fused(i: int) -> bool:
        return descs[i].get("shape","").upper() == "FUSED"

    # 1) 対角のみの最大連続区間
    i = 0
    while i < n:
        if not is_fused(i) and is_diag(i):
            j = i + 1
            while j < n and (not is_fused(j)) and is_diag(j):
                j += 1
            if j - i >= 2:
                length = (j - 1) - i + 1
                out.append((i, j - 1, cfg.fusion_score_diag_per_gate * float(length)))
            i = j
        else:
            i += 1

    # 2) 同軸1Q回転の連鎖
    def is_same_axis_1q(idx: int) -> bool:
        if is_fused(idx): return False
        d = descs[idx]; t = d["tag"].upper(); qs = d.get("qubits",[])
        return t in {"RX","RY","RZ"} and len(qs)==1

    i = 0
    while i < n:
        if is_same_axis_1q(i):
            axis = descs[i]["tag"].upper(); qb = descs[i]["qubits"][0]
            j = i + 1; ok = False
            while j < n and is_same_axis_1q(j) and descs[j]["tag"].upper()==axis and descs[j]["qubits"][0]==qb:
                ok = True; j += 1
            if ok:
                length = (j - 1) - i + 1
                out.append((i, j - 1, cfg.fusion_score_same_axis_per_gate * float(length)))
                i = j
            else:
                i += 1
        else:
            i += 1

    # 3) CX-RZ-CX
    i = 0
    while i + 2 < n:
        a, b, c = descs[i], descs[i+1], descs[i+2]
        if any(descs[k].get("shape","").upper()=="FUSED" for k in (i,i+1,i+2)):
            i += 1; continue
        if a["tag"].upper() in {"CX","CNOT"} and c["tag"].upper() in {"CX","CNOT"}:
            if a.get("qubits")==c.get("qubits") and b["tag"].upper()=="RZ":
                if len(b.get("qubits",[]))==1 and b["qubits"][0]==a["qubits"][1]:
                    out.append((i, i+2, cfg.fusion_score_phase_gadget))
                    i += 3; continue
        i += 1

    # 4) H-CX-H
    i = 0
    while i + 2 < n:
        h1, cx, h2 = descs[i], descs[i+1], descs[i+2]
        if any(descs[k].get("shape","").upper()=="FUSED" for k in (i,i+1,i+2)):
            i += 1; continue
        if h1["tag"].upper()=="H" and cx["tag"].upper() in {"CX","CNOT"} and h2["tag"].upper()=="H":
            if len(h1.get("qubits",[]))==1 and len(h2.get("qubits",[]))==1:
                if h1["qubits"][0]==cx["qubits"][1] and h2["qubits"][0]==cx["qubits"][1]:
                    out.append((i, i+2, cfg.fusion_score_hcxh))
                    i += 3; continue
        i += 1

    out.sort(key=lambda x: (x[0], -x[2]))
    return out