import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ..cfg import Config
from ..envs.qmdd_fusion_env import QMDDFusionEnv
from ..models.pointer_policy import PointerPolicy
from ..signatures import gate_to_signature_stub, signature_cost, simulate_update
from ..io.checkpoint import save_checkpoint, export_torchscript, export_onnx

def _teacher_choice(env: QMDDFusionEnv):
    best_i, best_delta = None, 1e9
    for i, g in enumerate(env.window):
        if g is None: continue
        gate_sig = gate_to_signature_stub(g.gate_type, g.acting_levels, env.cfg.top_k_levels)
        before = signature_cost(env.prefix_sig, env.cfg)
        after = signature_cost(simulate_update(env.prefix_sig, gate_sig, env.cfg.top_k_levels), env.cfg)
        delta = after - before
        if delta < best_delta:
            best_delta, best_i = delta, i
    return best_i if best_i is not None else 0

def run_supervised(cfg: Config, epochs: int = 200, batch_size: int = 1024,
                   save_dir: str = "ckpts", save_every: int = 20,
                   export_dir: str = "exports", export_final: bool = True):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    env = QMDDFusionEnv(cfg)
    model = PointerPolicy(cfg)
    opt = optim.Adam(model.parameters(), lr=3e-4)

    last_ckpt = None
    for ep in range(epochs):
        obs, _ = env.reset()
        Xsig, Xwin, Xmask, y = [], [], [], []
        for _ in range(batch_size):
            Xsig.append(obs["sig"]); Xwin.append(obs["win"]); Xmask.append(obs["mask"])
            a = _teacher_choice(env)
            y.append(a)
            obs, r, done, _, _ = env.step(a)
            if done: obs,_ = env.reset()

        Xsig = torch.tensor(np.stack(Xsig,0), dtype=torch.float32)
        Xwin = torch.tensor(np.stack(Xwin,0), dtype=torch.float32)
        Xmask= torch.tensor(np.stack(Xmask,0), dtype=torch.float32)
        y    = torch.tensor(np.array(y), dtype=torch.long)

        logits, _ = model({"sig":Xsig, "win":Xwin, "mask":Xmask})
        logits = logits + torch.log(Xmask + 1e-8)
        loss = nn.CrossEntropyLoss()(logits, y)

        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0:
            print(f"[SL] epoch={ep} loss={loss.item():.4f}")

        if (ep+1) % save_every == 0 or ep == epochs-1:
            ckpt_path = os.path.join(save_dir, f"pointer_policy_sl_ep{ep+1}.pt")
            save_checkpoint(model, cfg, ckpt_path, step=ep+1)
            last_ckpt = ckpt_path

    # 自動エクスポート（最後のチェックポイントで）
    if export_final and last_ckpt is not None:
        model_cpu = PointerPolicy(cfg)
        model_cpu.load_state_dict(torch.load(last_ckpt, map_location="cpu")["state_dict"])
        model_cpu.eval()
        ts_path = os.path.join(export_dir, f"pointer_policy_sl_ep{epochs}.ts.pt")
        onnx_path = os.path.join(export_dir, f"pointer_policy_sl_ep{epochs}.onnx")
        export_torchscript(model_cpu, ts_path)
        export_onnx(model_cpu, onnx_path, cfg.top_k_levels, cfg.window_size, cfg.sig_dim, cfg.gate_feat_dim)