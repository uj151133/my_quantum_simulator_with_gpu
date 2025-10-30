import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from typing import Optional
from ..cfg import Config
from ..envs.qmdd_fusion_env import QMDDFusionEnv
from ..models.pointer_policy import PointerPolicy
from ..io.checkpoint import save_checkpoint, load_checkpoint, export_torchscript, export_onnx

def _compute_adv(rew, val, done, gamma, lam):
    T = len(rew)
    adv = torch.zeros_like(rew)
    lastgaelam = 0.0
    nextv = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nextv * nonterm - val[t]
        lastgaelam = delta + gamma * lam * nonterm * lastgaelam
        adv[t] = lastgaelam
        nextv = val[t]
    ret = adv + val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret

def run_ppo(cfg: Config, save_dir: str = "ckpts", save_every_steps: int = 50_000,
            init_ckpt: Optional[str] = None, export_dir: str = "exports", export_final: bool = True):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    env = QMDDFusionEnv(cfg)
    policy = PointerPolicy(cfg).to(cfg.device)

    # ウォームスタート（模倣学習の重みを流用）
    if init_ckpt and os.path.isfile(init_ckpt):
        load_checkpoint(policy, init_ckpt, map_location=cfg.device)

    opt = optim.Adam(policy.parameters(), lr=cfg.lr)

    steps = 0
    last_ckpt = None
    while steps < cfg.total_steps:
        # rollout
        buf = {"sig":[], "win":[], "mask":[], "act":[], "logp":[], "rew":[], "val":[], "done":[]}
        obs,_ = env.reset()
        for _ in range(cfg.rollout_steps):
            o = {k: torch.tensor(v[None,...], dtype=torch.float32, device=cfg.device) for k,v in obs.items()}
            with torch.no_grad():
                logits, v = policy(o)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs=probs)
                a = dist.sample()
                logp = dist.log_prob(a)
            obs2, r, done, _, _ = env.step(int(a.item()))
            # store cpu
            buf["sig"].append(obs["sig"]); buf["win"].append(obs["win"]); buf["mask"].append(obs["mask"])
            buf["act"].append(a.item()); buf["logp"].append(logp.item()); buf["val"].append(v.item()); buf["rew"].append(r); buf["done"].append(done)
            obs = obs2
            if done: obs,_ = env.reset()

        # stack
        S = len(buf["act"])
        obs_sig = torch.tensor(np.stack(buf["sig"],0), dtype=torch.float32, device=cfg.device)
        obs_win = torch.tensor(np.stack(buf["win"],0), dtype=torch.float32, device=cfg.device)
        obs_mask= torch.tensor(np.stack(buf["mask"],0), dtype=torch.float32, device=cfg.device)
        act     = torch.tensor(np.array(buf["act"]), dtype=torch.long, device=cfg.device)
        old_logp= torch.tensor(np.array(buf["logp"]), dtype=torch.float32, device=cfg.device)
        val     = torch.tensor(np.array(buf["val"]), dtype=torch.float32, device=cfg.device)
        rew     = torch.tensor(np.array(buf["rew"]), dtype=torch.float32, device=cfg.device)
        done_t  = torch.tensor(np.array(buf["done"]).astype(np.float32), device=cfg.device)

        adv, ret = _compute_adv(rew, val, done_t, cfg.gamma, cfg.lam)

        # PPO updates
        idx = np.arange(S)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for i in range(0, S, cfg.minibatch_size):
                mb = idx[i:i+cfg.minibatch_size]
                logits, v_pred = policy({"sig":obs_sig[mb], "win":obs_win[mb], "mask":obs_mask[mb]})
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0-cfg.clip_eps, 1.0+cfg.clip_eps) * adv[mb]
                pg_loss = -(torch.min(surr1, surr2)).mean()
                v_loss = 0.5 * ((v_pred - ret[mb])**2).mean()
                ent = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef*v_loss - cfg.ent_coef*ent

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()

        steps += cfg.rollout_steps
        avgR = float(np.mean(buf["rew"])) if len(buf["rew"]) else 0.0
        print(f"[PPO] steps={steps} avgR={avgR:.3f}")

        if steps % save_every_steps == 0 or steps >= cfg.total_steps:
            ckpt_path = os.path.join(save_dir, f"pointer_policy_ppo_steps{steps}.pt")
            save_checkpoint(policy.cpu(), cfg, ckpt_path, step=steps)
            last_ckpt = ckpt_path
            policy.to(cfg.device)

    # 自動エクスポート（最後のチェックポイントで）
    if export_final and last_ckpt is not None:
        cpu_model = PointerPolicy(cfg)
        cpu_model.load_state_dict(torch.load(last_ckpt, map_location="cpu")["state_dict"])
        cpu_model.eval()
        ts_path = os.path.join(export_dir, f"pointer_policy_ppo_steps{steps}.ts.pt")
        onnx_path = os.path.join(export_dir, f"pointer_policy_ppo_steps{steps}.onnx")
        export_torchscript(cpu_model, ts_path)
        export_onnx(cpu_model, onnx_path, cfg.top_k_levels, cfg.window_size, cfg.sig_dim, cfg.gate_feat_dim)