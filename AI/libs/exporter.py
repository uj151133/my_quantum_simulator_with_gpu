import os
import torch
import onnx

SCHEDULER = "AI/exports/אָדָם.onnx"
FUSER     = "AI/exports/חַוָּה.onnx"

def _infer_in_dim(module: torch.nn.Module, default: int = 128) -> int:
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            return int(m.in_features)
    for _, p in module.named_parameters():
        if p.ndim == 2:  # [out, in]
            return int(p.shape[1])
    return default

def export_models(scheduler_model: torch.nn.Module, fuser_model: torch.nn.Module, out_dir: str = "AI/exports"):
    # ディレクトリ作成（固定名を使うため out_dir は無視）
    os.makedirs(os.path.dirname(SCHEDULER), exist_ok=True)
    os.makedirs(os.path.dirname(FUSER), exist_ok=True)

    # ASCII一時パスに保存 → 最後に固定名へ置換
    tmp_sched = os.path.join(os.path.dirname(SCHEDULER), "_sched_tmp.onnx")
    tmp_fuser = os.path.join(os.path.dirname(FUSER), "_fuser_tmp.onnx")

    # eval モード
    scheduler_model.eval()
    fuser_model.eval()

    D_sched = _infer_in_dim(scheduler_model, 128)
    D_fuser = _infer_in_dim(fuser_model, 128)

    # Scheduler: dynamo_export 優先、失敗時に旧export（両方とも一時パスへ保存）
    try:
        from torch.onnx import dynamo_export
        from torch.export import Dim
        N = 8
        x = torch.randn(N, D_sched)
        ep = dynamo_export(
            scheduler_model, x,
            dynamic_shapes={x: {0: Dim("N")}},  # バッチ(N)のみ動的
            export_options=torch.onnx.ExportOptions(opset_version=18),
        )
        # ep.save(tmp_sched)
        onnx.save_model(ep.model_proto, tmp_sched)
    except Exception:
        x = torch.randn(8, D_sched)
        torch.onnx.export(
            scheduler_model, x, tmp_sched,
            opset_version=18, input_names=["input"], output_names=["logits"]
        )
    # 一時→固定名（非ASCII）へアトミック置換
    os.replace(tmp_sched, SCHEDULER)

    # Fuser: [D] を基本。一時パスに保存してから移動
    try:
        from torch.onnx import dynamo_export
        x = torch.randn(D_fuser)
        ep = dynamo_export(
            fuser_model, x,
            export_options=torch.onnx.ExportOptions(opset_version=18),
        )
        # ep.save(tmp_fuser)
        onnx.save_model(ep.model_proto, tmp_fuser)
    except Exception:
        try:
            x = torch.randn(1, D_fuser)
            torch.onnx.export(
                fuser_model, x, tmp_fuser,
                opset_version=18, input_names=["input"], output_names=["fused"]
            )
        except Exception:
            x = torch.randn(D_fuser)
            torch.onnx.export(
                fuser_model, x, tmp_fuser,
                opset_version=18, input_names=["input"], output_names=["fused"]
            )
    os.replace(tmp_fuser, FUSER)

    # 学習重み（.pt）も固定名ベースで保存
    torch.save(scheduler_model.state_dict(), os.path.splitext(SCHEDULER)[0] + ".pt")
    torch.save(fuser_model.state_dict(),     os.path.splitext(FUSER)[0] + ".pt")
    print(f"[EXPORT] scheduler -> {SCHEDULER}")
    print(f"[EXPORT] fuser     -> {FUSER}")