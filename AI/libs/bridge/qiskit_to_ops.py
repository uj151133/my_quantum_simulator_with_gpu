from __future__ import annotations
from typing import List, Tuple, Dict, Any

def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0  # 未束縛パラメータは 0.0 にフォールバック（必要なら事前 bind）

# Qiskit命名 → シミュレータ側 gate_type へ正規化
_NAME_MAP = {
    "id": "ID",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "h": "H",
    "s": "S",
    "sdg": "SDG",
    "t": "T",
    "tdg": "TDG",
    "sx": "SX",   # C++では addV に対応させる
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "p":  "P",
    "u":  "U3",   # Qiskit の U は一般に U3(θ,φ,λ)
    "u1": "U1",
    "u2": "U2",
    "u3": "U3",
    "cx": "CX",
    "cy": "CY",
    "cz": "CZ",
    "cp": "CP",
    "ch": "CH",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "swap": "SWAP",
}

# 対角ゲートの簡易判定（必要に応じて拡張）
_DIAG = {"Z", "RZ", "P", "U1", "CZ", "CP"}

def circuit_to_ops(qc) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Qiskit QuantumCircuit -> (num_qubits, ops list)
    ops 要素:
      {"gate_type":str, "qubits":[...], "theta":float, "phi":float, "lam":float, "is_diag":int}
    """
    n = qc.num_qubits
    ops: List[Dict[str, Any]] = []
    for inst, qargs, _ in qc.data:
        name = inst.name.lower()
        if name not in _NAME_MAP:
            continue
        g = _NAME_MAP[name]

        qubits = [qb.index for qb in qargs]
        theta = phi = lam = 0.0
        params = getattr(inst, "params", [])

        if g in ("RX", "RY", "RZ", "P", "U1", "CRX", "CRY", "CRZ", "CP"):
            if len(params) >= 1: theta = _to_float(params[0])
        if g in ("U2",):
            if len(params) >= 1: phi = _to_float(params[0])
            if len(params) >= 2: lam = _to_float(params[1])
        if g in ("U3", "U"):
            if len(params) >= 1: theta = _to_float(params[0])
            if len(params) >= 2: phi   = _to_float(params[1])
            if len(params) >= 3: lam   = _to_float(params[2])

        is_diag = 1 if g in _DIAG else 0

        ops.append({
            "gate_type": g,
            "qubits": qubits,
            "theta": float(theta),
            "phi":   float(phi),
            "lam":   float(lam),
            "is_diag": int(is_diag),
        })
    return n, ops