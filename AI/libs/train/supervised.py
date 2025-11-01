import os, time, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ..cfg import Config
from ..envs.qmdd_scheduling_env import QMDDFusionEnv
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
    model = PointerPolicy(cfg).to(cfg.device)
    opt = optim.Adam(model.parameters(), lr=3e-4)

    last_ckpt = None
    for ep in range(epochs):
        t0 = time.time()
        obs, _ = env.reset()
        Xsig, Xwin, Xmask, y = [], [], [], []
        for _ in range(batch_size):
            Xsig.append(obs["sig"]); Xwin.append(obs["win"]); Xmask.append(obs["mask"])
            a = _teacher_choice(env)
            y.append(a)
            obs, r, done, _, _ = env.step(a)
            if done: obs,_ = env.reset()

        # numpy -> torch

        Xsig = torch.tensor(np.stack(Xsig,0), dtype=torch.float32, device=cfg.device)
        Xwin = torch.tensor(np.stack(Xwin,0), dtype=torch.float32, device=cfg.device)
        Xmask= torch.tensor(np.stack(Xmask,0), dtype=torch.float32, device=cfg.device)
        y    = torch.tensor(np.array(y), dtype=torch.long, device=cfg.device)

        model.train()
        opt.zero_grad()
        logits, _ = model({"sig": Xsig, "win": Xwin, "mask": Xmask})
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        opt.step()
        t1 = time.time()
        if ep % 10 == 0:
            print(f"[SL] epoch={ep} loss={loss.item():.4f} time/epoch={(t1-t0)*1000:.1f}ms")

        if (ep+1) % save_every == 0 or ep == epochs-1:
            ckpt_path = os.path.join(save_dir, f"pointer_policy_sl_ep{ep+1}.pt")
            save_checkpoint(model.cpu(), cfg, ckpt_path, step=ep+1)
            last_ckpt = ckpt_path
            model.to(cfg.device)

    if export_final and last_ckpt is not None:
        cpu_model = PointerPolicy(cfg)
        cpu_model.load_state_dict(torch.load(last_ckpt, map_location="cpu")["state_dict"])
        cpu_model.eval()
        ts_path = os.path.join(export_dir, f"pointer_policy_sl_ep{epochs}.ts.pt")
        onnx_path = os.path.join(export_dir, f"pointer_policy_sl_ep{epochs}.onnx")
        export_torchscript(cpu_model, ts_path)
        export_onnx(cpu_model, onnx_path, cfg.top_k_levels, cfg.window_size, cfg.sig_dim, cfg.gate_feat_dim)