import argparse
from AI.libs.parameter import Parameter
from AI.libs.circuit_gen import make_random_circuits
from AI.libs.scheduler import SchedulerModel, train_supervised_scheduler, train_ppo_scheduler
from AI.libs.fuser import FuserModel, train_supervised_fuser, train_ppo_fuser
from AI.libs.exporter import export_models
from AI.libs.eval_speed import evaluate_policy_speedup_for_qiskit
from AI.libs.bridge import Bridge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--algo", choices=["sup", "ppo"], default="sup")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max_outer_iters", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    params = Parameter.load()
    if args.device: params.general.device = args.device
    params.general.window_size = args.window

    in_dim = params.general.gate_feat_dim + params.general.sig_dim * params.general.top_k_levels
    bridge = Bridge()

    if args.mode == "train":
        train_qcs = make_random_circuits(num=20, num_qubits=6, depth=80)
        sched = SchedulerModel(input_dim=in_dim, hidden=128, use_transformer=False, device=params.general.device)
        fuse  = FuserModel(input_dim=in_dim,  hidden=128, use_transformer=False, device=params.general.device)
        if args.algo == "sup":
            train_supervised_scheduler(sched, train_qcs, params, epochs=5)
            train_supervised_fuser(fuse, train_qcs, params, epochs=5)
        else:
            train_ppo_scheduler(sched, bridge, train_qcs, params, episodes=args.episodes)
            train_ppo_fuser(fuse, bridge, train_qcs, params, episodes=args.episodes)
        export_models(sched, fuse)
        # エクスポート直後に速度比較も自動実行（fusion セッションは使わず pre-stage 比較）
        eval_qcs = make_random_circuits(num=3, num_qubits=10, depth=200)
        for i, qc in enumerate(eval_qcs):
            print(f"=== Speed Eval after export: Circuit {i} ===")
            evaluate_policy_speedup_for_qiskit(
                qc=qc,
                bridge=bridge,
                scheduler_model=sched,   # 学習直後のモデル
                params=params,
                trials=args.trials,
                max_outer_iters=args.max_outer_iters,
            )
    else:
        eval_qcs = make_random_circuits(num=3, num_qubits=6, depth=60)
        sched = SchedulerModel(input_dim=in_dim, hidden=128, use_transformer=False, device=params.general.device)
        fuse  = FuserModel(input_dim=in_dim,  hidden=128, use_transformer=False, device=params.general.device)
        for i, qc in enumerate(eval_qcs):
            print(f"=== Circuit {i} ===")
            def session_factory(nq: int, ops):
                return bridge.new_session(nq, ops)
            def runtime_eval(session) -> float:
                return bridge.evaluate_session_runtime(session)
            evaluate_policy_speedup_for_qiskit(
                qc=qc,
                bridge=bridge,
                scheduler_model=sched,
                session_factory=session_factory,
                runtime_eval=runtime_eval,
                fusion_model=fuse,
                params=params,
                trials=args.trials,
                max_outer_iters=args.max_outer_iters,
            )

if __name__ == "__main__":
    main()