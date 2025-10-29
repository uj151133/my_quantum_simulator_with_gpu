import time, random
from typing import List, Dict, Any

class CppClient:
    """
    C++ QMDD へのブリッジ（今はスタブ）。
    後で pybind11 / RPC に置き換え。
    """
    def evaluate_chunk_runtime(self, circuit_id: str, chunk_ops: List[Dict[str,Any]]) -> Dict[str, Any]:
        # ダミー: 非対角が多いほど遅いというモデル
        time.sleep(0.001)
        n_nondiag = sum(1 for op in chunk_ops if op.get("is_diag",0)==0)
        wall = 0.05 * n_nondiag + 0.001 * random.random()
        nodes_peak = 100 + 10 * n_nondiag
        return {"wall_time_ms": wall*1000.0, "nodes_peak": nodes_peak, "nodes_delta": 10*n_nondiag}