from typing import List, Dict, Any

def pointer_policy_schedule(ops: List[Dict[str,Any]], model, cfg) -> List[Dict[str,Any]]:
    """
    ops: [{"tag": "RZ", "qubits":[...], "theta":..., "is_diag":...}, ...] など Core 相当の辞書
    model: 既存 PointerPolicy
    戻り: 並べ替え後の ops
    """
    # ここは既存コードで構築している ready 集合＋mask を流用して、
    # model から一つずつ取り出す方式に置き換えてください。
    # 例示だけ置きます（実装詳細はあなたのモデルの forward に合わせて差し替え）
    order = []
    used = set()
    # トポソートを守った上で、モデルのスコア順に取り出す処理…
    # TODO: 実装を既存スケジューラから移植
    return ops  # 並べ替え後を返す