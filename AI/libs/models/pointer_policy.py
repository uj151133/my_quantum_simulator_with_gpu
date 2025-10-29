import torch
import torch.nn as nn

class PointerPolicy(nn.Module):
    """
    位置ポインタ型の方策。
    入力: sig[K,sd], win[W,F], mask[W]
    出力: logits[W], value
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = 128
        nhead = 8
        nl = 3

        self.sig_proj = nn.Linear(cfg.sig_dim * cfg.top_k_levels, D)
        self.win_proj = nn.Linear(cfg.gate_feat_dim, D)
        enc_layer = nn.TransformerEncoderLayer(d_model=D, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nl)

        self.pointer = nn.Linear(D, 1)
        self.value_head = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs):
        sig = obs["sig"]        # [B,K,sd]
        win = obs["win"]        # [B,W,F]
        mask = obs["mask"]      # [B,W]

        B = sig.shape[0]
        sig_flat = sig.reshape(B, -1)               # [B, K*sd]
        sig_emb = self.sig_proj(sig_flat)[:,None,:] # [B,1,D]
        win_emb = self.win_proj(win)                # [B,W,D]
        x = torch.cat([sig_emb, win_emb], dim=1)    # [B,1+W,D]
        enc = self.encoder(x)
        win_enc = enc[:,1:,:]                       # [B,W,D]
        logits = self.pointer(win_enc).squeeze(-1)  # [B,W]
        logits = logits + torch.log(mask + 1e-8)    # マスク
        value = self.value_head(enc[:,0,:]).squeeze(-1)
        return logits, value