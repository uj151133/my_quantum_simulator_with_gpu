import os
import torch
from dataclasses import asdict
from typing import Optional, Dict, Any

def save_checkpoint(model: torch.nn.Module, cfg, path: str, step: Optional[int]=None, extra: Optional[Dict[str,Any]]=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "cfg": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else None,
        "step": step,
        "extra": extra or {},
    }
    torch.save(payload, path)
    print(f"[ckpt] saved: {path}")

def load_checkpoint(model: torch.nn.Module, path: str, map_location: Optional[str]="cpu") -> Dict[str,Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    print(f"[ckpt] loaded weights from: {path}")
    return payload

def export_torchscript(model: torch.nn.Module, ts_path: str):
    """
    forward_tensors(sig, win, mask) をエントリにするラッパーをscript。
    失敗したらtraceでフォールバック。
    """
    class Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner
        def forward(self, sig: torch.Tensor, win: torch.Tensor, mask: torch.Tensor):
            logits, value = self.inner.forward_tensors(sig, win, mask)
            return logits, value

    os.makedirs(os.path.dirname(ts_path), exist_ok=True)
    model_eval = Wrapper(model).eval()

    # script を試し、失敗時は trace にフォールバック
    try:
        scripted = torch.jit.script(model_eval)
        scripted.save(ts_path)
        print(f"[export] TorchScript saved: {ts_path}")
    except Exception as e:
        print(f"[export] TorchScript script failed ({e}), fallback to trace.")
        # 形状は cfg から取得
        cfg = getattr(model, "cfg")
        sig  = torch.zeros(1, cfg.top_k_levels, cfg.sig_dim, dtype=torch.float32)
        win  = torch.zeros(1, cfg.window_size, cfg.gate_feat_dim, dtype=torch.float32)
        mask = torch.ones (1, cfg.window_size,               dtype=torch.float32)
        traced = torch.jit.trace(model_eval, (sig, win, mask))
        traced.save(ts_path)
        print(f"[export] TorchScript (traced) saved: {ts_path}")

def export_onnx(model: torch.nn.Module, onnx_path: str, K: int, W: int, sd: int, fd: int, opset: int = 18) -> bool:
    """
    ONNXエクスポート。forward_tensors を使い、動的バッチ軸を設定。
    onnx / onnxscript が無い場合はスキップして False を返す。
    """
    # 依存チェック
    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
    except ModuleNotFoundError as e:
        pkg = "onnxscript" if "onnxscript" in str(e) else "onnx"
        print(f"[export][skip] {pkg} not installed. Run: pip install onnx onnxscript onnxruntime")
        return False

    class Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner
        def forward(self, sig: torch.Tensor, win: torch.Tensor, mask: torch.Tensor):
            logits, value = self.inner.forward_tensors(sig, win, mask)
            return logits, value

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model_eval = Wrapper(model).eval()

    sig  = torch.zeros(1, K, sd, dtype=torch.float32)
    win  = torch.zeros(1, W, fd, dtype=torch.float32)
    mask = torch.ones (1, W,     dtype=torch.float32)

    dynamic_axes = {
        "sig":    {0: "batch"},
        "win":    {0: "batch"},
        "mask":   {0: "batch"},
        "logits": {0: "batch"},
        "value":  {0: "batch"},
    }
    input_names  = ["sig", "win", "mask"]
    output_names = ["logits", "value"]

    try:
        torch.onnx.export(
            model_eval, (sig, win, mask), onnx_path,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True
        )
        print(f"[export] ONNX saved: {onnx_path}")
        return True
    except Exception as e:
        print(f"[export][fail] ONNX export failed: {e}")
        return False