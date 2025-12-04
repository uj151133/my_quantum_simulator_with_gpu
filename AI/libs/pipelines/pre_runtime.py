from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from AI.libs.scheduler import SchedulerModel, reorder_ops_by_model
from AI.libs.parameter import Parameter
from AI.libs.bridge import Bridge


class PreRuntime:
    """
    IR 段階の最適化パイプライン（完全版）
    - law（C++ optimize_ops）と schedule（Python モデルによる順序入替）を順不同に試し、計測して良い方を採択
    - 並べ替え実変換は、C++の reorder_ops があればそちらを使う設計にできるが、ここでは Python 実装を既定
    """
    def __init__(
        self,
        nq: int,
        model: Optional[SchedulerModel],
        params: Parameter,
        trials: int = 2,
        max_iters: int = 4,
        improve_eps: float = 0.01,
        device: str = "cpu",
        bridge: Bridge | None = None
    ):
        self.bridge = bridge or Bridge()
        self.nq = nq
        self.model = model
        self.params = params
        self.trials = trials
        self.max_iters = max_iters
        self.improve_eps = improve_eps
        self.device = device

    def _measure_ms(self, ops: List[Dict[str, Any]], label: str) -> float:
        vals = []
        for _ in range(self.trials):
            res = self.bridge.evaluate_runtime_for_ops(self.nq, ops, label)
            vals.append(float(res["wall_time_ms"]))
        return float(np.mean(vals)) if vals else float("inf")

    def _schedule(self, ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.model is None:
            return ops
        return reorder_ops_by_model(
            ops, self.model, params=self.params, device=self.device,
            bridge=self.bridge, use_cpp_dag=True
        )

    def run(self, ops_in: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        best_ops = ops_in
        best_ms = self._measure_ms(best_ops, label="pre0")

        for it in range(self.max_iters):
            # A: law -> schedule
            ops_a = self.bridge.optimize_ops(best_ops)
            ops_a = self._schedule(ops_a)
            ms_a = self._measure_ms(ops_a, label=f"preA{it}")

            # B: schedule -> law
            ops_b = self._schedule(best_ops)
            ops_b = self.bridge.optimize_ops(ops_b)
            ms_b = self._measure_ms(ops_b, label=f"preB{it}")

            cand_ops, cand_ms = (ops_a, ms_a) if ms_a <= ms_b else (ops_b, ms_b)
            if best_ms - cand_ms < self.improve_eps:
                break
            best_ops, best_ms = cand_ops, cand_ms

        return best_ops, best_ms