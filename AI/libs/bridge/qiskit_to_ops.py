# Qiskit circuit -> ops(list[dict]) 変換（Qiskit 2.x 対応）
from __future__ import annotations

from typing import List, Tuple, Dict, Any

try:
    # 新系（推奨）
    from qiskit.circuit import QuantumCircuit, Qubit
except Exception:  # フォールバック（環境によっては __init__ で再エクスポートされない）
    from qiskit.circuit.quantumcircuit import QuantumCircuit  # type: ignore
    from qiskit.circuit import Qubit  # type: ignore


def _qubit_index(circ: QuantumCircuit, qb: Qubit) -> int:
    # Qiskit 2.x 推奨
    try:
        return circ.find_bit(qb).index  # type: ignore[attr-defined]
    except Exception:
        # フォールバック（古い挙動に近い）
        return circ.qubits.index(qb)


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        # ParameterExpression など
        v = getattr(x, "value", None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    # 最後の手段（未解決パラメータ等）
    return 0.0


def circuit_to_ops(circ: QuantumCircuit) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Convert a Qiskit QuantumCircuit to a list of ops dictionaries that CppClient understands.
    Returns:
      (num_qubits, ops)
    Each op is a dict like:
      { "gate_type": "H"|"CX"|"RZ"|..., "qubits": [int,...], "theta": float, "phi": float, "lam": float }
    """
    n_qubits = len(circ.qubits)
    ops: List[Dict[str, Any]] = []

    for item in circ.data:
        # Qiskit 2.x: item は CircuitInstruction
        # 旧系: (operation, qargs, cargs) のタプル
        if hasattr(item, "operation"):
            operation = item.operation
            qargs = item.qubits
        else:
            operation, qargs, _cargs = item  # type: ignore[misc]

        name = getattr(operation, "name", "").lower()
        # 非ユニタリはスキップ
        if name in ("measure", "barrier", "delay", "snapshot", "reset"):
            continue

        qidxs = [_qubit_index(circ, qb) for qb in qargs]
        params = getattr(operation, "params", []) or []

        op: Dict[str, Any] = {"gate_type": name.upper(), "qubits": qidxs}

        # 代表的な回転系
        if name in ("rz", "rx", "ry"):
            if len(params) >= 1:
                op["theta"] = _as_float(params[0])
        # U, U3（Qiskit 2.x の UGate は name=="u"）
        elif name in ("u", "u3"):
            if len(params) >= 3:
                op["theta"] = _as_float(params[0])
                op["phi"] = _as_float(params[1])
                op["lam"] = _as_float(params[2])
            op["gate_type"] = "U3"

        # そのまま渡す代表例（実装側が対応している想定）
        # X, H, CX, CZ, SX などは gate_type=上記でOK

        # 未知ゲート名の扱い（必要ならここでマッピング/分解）
        # 例: 'sx' を RX(pi/2) に分解 など
        # 今はそのまま C++ 側に委譲
        ops.append(op)

    return n_qubits, ops