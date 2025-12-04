import argparse, torch
from AI.libs.parameter import Parameter
from AI.libs.models.pointer_policy import PointerPolicy
from AI.libs.io.checkpoint import load_checkpoint, export_torchscript, export_onnx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    ap.add_argument("--ts_out", default=None, help="output TorchScript .pt path")
    ap.add_argument("--onnx_out", default=None, help="output ONNX .onnx path")
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--top_k_levels", type=int, default=8)
    ap.add_argument("--sig_dim", type=int, default=6)
    ap.add_argument("--gate_feat_dim", type=int, default=16)
    args = ap.parse_args()

    params = Parameter(window_size=args.window_size, top_k_levels=args.top_k_levels, sig_dim=args.sig_dim, gate_feat_dim=args.gate_feat_dim)
    model = PointerPolicy(params)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model.eval()

    # ダミー入力（バッチ1）
    sig = torch.zeros(1, params.general.top_k_levels, params.general.sig_dim, dtype=torch.float32)
    win = torch.zeros(1, params.general.window_size, params.general.gate_feat_dim, dtype=torch.float32)
    mask= torch.ones(1, params.general.window_size, dtype=torch.float32)

    example = {"sig": sig, "win": win, "mask": mask}

    if args.ts_out:
        export_torchscript(model, example, args.ts_out)
    if args.onnx_out:
        export_onnx(model, example, args.onnx_out, opset=18)

if __name__ == "__main__":
    main()