from typing import List, Tuple, Any, Dict
import numpy as np

class PreSimOptimizer:
    def __init__(self, bridge, model_order, cfg, max_outer_iters=3, improve_eps=0.01):
        self.bridge = bridge    # qmdd_bridge
        self.model_order = model_order
        self.cfg = cfg
        self.max_outer = max_outer_iters
        self.eps = improve_eps

    def measure_ms(self, cpp_client, ops, trials=2) -> float:
        vals=[]
        for _ in range(trials):
            res = cpp_client.evaluate_chunk_runtime("pre", ops)
            vals.append(float(res["wall_time_ms"]))
        return float(np.mean(vals))

    def reorder_with_model(self, ops):
        # ここで pointer_policy_schedule を呼ぶ
        return ops

    def run(self, cpp_client, ops_in: List[Dict[str,Any]]) -> Tuple[List[Dict[str,Any]], float]:
        best_ops = ops_in
        best_ms = self.measure_ms(cpp_client, best_ops)
        for it in range(self.max_outer):
            ops_law = self.bridge.optimize_ops(best_ops)   # law（C++）で最適化
            ops_ord = self.reorder_with_model(ops_law)     # 並べ替えモデル
            ms = self.measure_ms(cpp_client, ops_ord)
            if best_ms - ms < self.eps:
                break
            best_ops, best_ms = ops_ord, ms
        return best_ops, best_ms