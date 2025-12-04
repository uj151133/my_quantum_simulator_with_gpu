from typing import List, Dict, Any, Tuple, Optional
from AI.libs.fusion_rules import score_candidates
from AI.libs.parameter import Parameter

class PostRuntime:
    """
    実行段階: 窓→（ルールスコア＋モデル提案）→非重複化→融合→step を繰り返し
    ルールスコアとモデル提案ボーナスは Parameter 駆動。
    """
    def __init__(self, bridge, model_fusion=None, window: int = 32, topk: Optional[int]=None, params: Optional[Parameter]=None):
        self.bridge = bridge
        self.model_fusion = model_fusion
        self.window = window
        self.topk = topk
        self.params = params or Parameter().load()

    def _merge_candidates(self, descs: List[Dict[str,Any]]) -> List[Tuple[int,int]]:
        scored = score_candidates(descs, params=self.params)  # [(s,e,score)]
        scored.sort(key=lambda x: (-x[2], x[0]))
        if self.model_fusion is not None:
            try:
                # params を渡す版に統一
                m_ranges = self.model_fusion.propose(descs, params=self.params)  # [(s,e)]
                for (s,e) in m_ranges or []:
                    scored.append((s,e,self.params.fusion_model_bonus))
            except Exception:
                pass
        taken: List[Tuple[int,int]] = []
        covered = [False]*len(descs)
        for s,e,_sc in sorted(scored, key=lambda x: (-x[2], x[0])):
            if s<0 or e<0 or s>=len(descs) or e>=len(descs) or s>e: continue
            if any(covered[k] for k in range(s,e+1)): continue
            taken.append((s,e))
            for k in range(s,e+1): covered[k] = True
            if self.topk is not None and len(taken) >= self.topk: break
        return taken

    def run(self, session) -> None:
        while True:
            window_desc = self.bridge.snapshot_queue_window(session, self.window)
            if not window_desc: break
            ranges = self._merge_candidates(window_desc)
            if ranges: self.bridge.propose_fusion(session, ranges)
            cont = self.bridge.step(session)
            if not cont: break