from typing import List, Dict, Any, Tuple
import numpy as np

class PreSimOptimizer:
    def __init__(self, cpp_client, model_order, cfg, max_outer_iters=3, improve_eps=0.01):
        self.cpp = cpp_client
        self.model_order = model_order  # 並べ替えモデル（Head A）
        self.cfg = cfg
        self.max_outer = max_outer_iters
        self.eps = improve_eps

    def measure_ms(self, ops: List[Dict[str,Any]], trials=2) -> float:
        vals = []
        for _ in range(trials):
            res = self.cpp.evaluate_chunk_runtime("pre_sim", ops)
            vals.append(float(res["wall_time_ms"]))
        return float(np.mean(vals))

    def reorder_with_model(self, ops: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        # ここは既存のポインタポリシーで ops の順序を決める関数を呼んでください
        # 例: return pointer_policy_schedule(ops, self.model_order, self.cfg)
        return ops  # ダミー（実装差し替え）

    def run(self, ops_in: List[Dict[str,Any]]) -> Tuple[List[Dict[str,Any]], float]:
        best_ops = ops_in
        best_ms = self.measure_ms(best_ops)
        for it in range(self.max_outer):
            # law（C++に最適化だけさせてopsを返すAPIが必要）
            ops_law = self.cpp.optimize_ops(best_ops, options_env=None)  # envでR1/R7=0などを指定可能
            # 並べ替え
            ops_ord = self.reorder_with_model(ops_law)
            ms = self.measure_ms(ops_ord)
            if best_ms - ms < self.eps:
                break
            best_ops, best_ms = ops_ord, ms
        return best_ops, best_ms