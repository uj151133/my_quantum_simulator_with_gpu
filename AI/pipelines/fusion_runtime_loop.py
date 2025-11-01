from typing import List, Dict, Any, Tuple

class FusionRuntime:
    def __init__(self, bridge, model_fusion, window=32):
        self.bridge = bridge      # qmdd_bridge
        self.model_fusion = model_fusion
        self.window = window

    def propose(self, descs: List[Dict[str,Any]]) -> List[Tuple[int,int]]:
        # descs は Core 相当の dict の配列
        # 例: 対角のみの連続区間をできるだけ長く fuse する簡単ルール
        out=[]
        s=None
        for i,d in enumerate(descs):
            if d["is_diag"]:
                if s is None: s=i
            else:
                if s is not None and i-1> s: out.append((s, i-1))
                s=None
        if s is not None and (len(descs)-1)>s: out.append((s, len(descs)-1))
        return out

    def run(self, session):
        while True:
            window_desc = self.bridge.snapshot_queue_window(session, self.window)
            if not window_desc:
                break
            ranges = self.propose(window_desc)
            if ranges:
                self.bridge.propose_fusion(session, ranges)
            cont = self.bridge.step(session)
            if not cont:
                break