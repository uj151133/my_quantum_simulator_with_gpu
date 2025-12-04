from typing import List, Dict, Any, Tuple, Optional
import time
import numpy as np
from qiskit.circuit import QuantumCircuit
from AI.libs.qiskit_to_core import circuit_to_core_list
from AI.libs.pipelines.pre_runtime import PreRuntime
from AI.libs.pipelines.post_runtime import PostRuntime
from AI.libs.scheduler import SchedulerModel
from AI.libs.parameter import Parameter


def _measure_ms(bridge, nq: int, ops, label: str, trials: int) -> float:
    # 1回ウォームアップ + 複数回の中央値
    bridge.evaluate_runtime_for_ops(nq, ops, label + "_warmup")
    vals = []
    for _ in range(trials):
        r = bridge.evaluate_runtime_for_ops(nq, ops, label)
        vals.append(float(r["wall_time_ms"]))
    return float(np.median(vals)) if vals else float("inf")

def evaluate_policy_speedup_for_qiskit(
    qc: QuantumCircuit,
    bridge,
    scheduler_model: Optional[SchedulerModel] = None,
    session_factory=None,
    runtime_eval=None,
    fusion_model=None,
    params: Parameter = Parameter.load(),
    trials: int = 3,
    max_outer_iters: int = 4,
) -> Tuple[int, float, float, float, float]:
    nq, cores = circuit_to_core_list(qc)
    # ops = [c.__dict__ for c in cores]
    ops = list(cores)
    # baseline_ms = float(bridge.evaluate_runtime_for_ops(nq, ops, "baseline")["wall_time_ms"])
    baseline_ms = _measure_ms(bridge, nq, ops, "baseline", trials)
    # pre = PreRuntime(bridge, nq, scheduler_model, device=params.general.device, trials=trials, max_iters=max_outer_iters, improve_eps=0.01, params=params)
    # pre = PreRuntime(
    #     nq=nq,
    #     model=scheduler_model,
    #     params=params,
    #     trials=trials,
    #     max_iters=max_outer_iters,
    #     improve_eps=0.01,
    #     device=params.general.device,
    #     bridge=bridge,
    # )
    # best_ops, presim_ms = pre.run(ops)
    pre = PreRuntime(
        nq=nq, model=scheduler_model, params=params,
        trials=trials, max_iters=max_outer_iters, improve_eps=0.01,
        device=getattr(params, "device", "cpu"), bridge=bridge,
    )
    t0 = time.perf_counter()
    best_ops, sim_ms = pre.run(ops)   # sim_ms は「並べ替え後のシミュレータ時間」
    e2e_ms = (time.perf_counter() - t0) * 1000.0  # 推論+DAG合法化+シミュレータの合計
    presim_ms = e2e_ms
    fusion_ms = presim_ms
    if session_factory and runtime_eval:
        try:
            session = session_factory(nq, best_ops)
            pr = PostRuntime(bridge=bridge, model_fusion=fusion_model, window=params.general.window_size, topk=None, params=params)
            pr.run(session)
            fusion_ms = float(runtime_eval(session))
        except Exception as e:
            print(f"[WARN] fusion stage skipped: {e}")
    delta_law_order = baseline_ms - presim_ms
    delta_fusion    = presim_ms - fusion_ms
    speedup = (baseline_ms - fusion_ms) / baseline_ms if baseline_ms > 0 else 0.0
    print(f"baseline:      {baseline_ms:.3f} ms")
    print(f"law+order:     {presim_ms:.3f} ms (Δ={delta_law_order:+.3f} ms)")
    print(f"runtime fusion:{fusion_ms:.3f} ms (Δ={delta_fusion:+.3f} ms)")
    print(f"total speedup: {speedup*100:.2f} %")
    return nq, baseline_ms, presim_ms, fusion_ms, speedup