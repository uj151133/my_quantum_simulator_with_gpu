from typing import List, Dict, Any
import torch
from AI.libs.config import Config

def build_signature(ops: List[Dict[str,Any]], cfg: Config) -> torch.Tensor:
    """
    簡易シグネチャ:
      - shape 分布ヒスト（DIAG/PERM/GENERAL/ANTI/FUSED）→ 5 次元
      - タグの出現頻度上位K（top_k_levels）を hash→bin して sig_dim-5 の次元に詰める
    出力: 長さ sig_dim * top_k_levels のベクトル（Kレベル分を縦に連結）
    """
    K = cfg.top_k_levels
    D = cfg.sig_dim
    out = torch.zeros(K * D)

    # 形カテゴリのインデックス
    def shape_idx(shape: str) -> int:
        s = (shape or "").upper()
        if s == "DIAG": return 0
        if s == "PERM": return 1
        if s == "GENERAL": return 2
        if s == "ANTI": return 3
        if s == "FUSED": return 4
        return 2  # default GENERAL

    # カウント
    shape_counts = [0,0,0,0,0]
    tag_counts: Dict[str,int] = {}
    for o in ops:
        shape = o.get("shape","GENERAL")
        shape_counts[shape_idx(shape)] += 1
        tag = (o.get("tag","") or "").upper()
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # 上位タグ K
    top_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))[:K]

    # レベルごとに D次元を書き込む
    for level in range(K):
        offset = level * D

        # 先頭5次元: shapeヒストを正規化して書く
        total = sum(shape_counts) or 1
        for i in range(5):
            out[offset + i] = float(shape_counts[i]) / float(total)

        # 残り (D-5) 次元: そのレベルのタグ名をハッシュしてビンを1つ立てる
        if level < len(top_tags):
            tag, _cnt = top_tags[level]
            bins = max(1, D - 5)
            idx = (hash(tag) % bins)
            out[offset + 5 + idx] = 1.0

    return out