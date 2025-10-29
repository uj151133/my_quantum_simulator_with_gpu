import os
import torch
from dataclasses import asdict
from typing import Optional, Dict, Any

def save_checkpoint(model: torch.nn.Module, cfg, path: str, step: Optional[int]=None, extra: Optional[Dict[str,Any]]=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "cfg": asdict(cfg) if hasattr(cfg, "__dict__") or hasattr(cfg, "__dataclass_fields__") else None,
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

def export_torchscript(model: torch.nn.Module, example_inputs: Dict[str, torch.Tensor], ts_path: str):
    """
    model.forward は obs(dict) を受けているので、薄いラッパで (sig, win, mask) を受ける形にします。
    """
    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, sig: torch.Tensor, win: torch.Tensor, mask: torch.Tensor):
            obs = {"sig": sig, "win": win, "mask": mask}
            logits, value = self.inner(obs)
            return logits, value

    model_eval = Wrapper(model).eval()
    with torch.no_grad():
        scripted = torch.jit.script(model_eval)
        scripted.save(ts_path)
    print(f"[export] TorchScript saved: {ts_path}")

def export_onnx(model: torch.nn.Module, example_inputs: Dict[str, torch.Tensor], onnx_path: str, opset: int = 18):
    """
    ONNXエクスポート。動的軸（バッチ次元）を設定。
    """
    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, sig: torch.Tensor, win: torch.Tensor, mask: torch.Tensor):
            obs = {"sig": sig, "win": win, "mask": mask}
            logits, value = self.inner(obs)
            return logits, value

    model_eval = Wrapper(model).eval()
    sig = example_inputs["sig"]
    win = example_inputs["win"]
    mask = example_inputs["mask"]

    dynamic_axes = {
        "sig": {0: "batch"},
        "win": {0: "batch"},
        "mask": {0: "batch"},
        "logits": {0: "batch"},
        "value": {0: "batch"},
    }
    input_names = ["sig", "win", "mask"]
    output_names = ["logits", "value"]

    torch.onnx.export(
        model_eval,
        (sig, win, mask),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[export] ONNX saved: {onnx_path}")