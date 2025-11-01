from typing import List, Dict, Any, Tuple

class FusionRuntime:
    def __init__(self, cpp_client, model_fusion, window=32):
        self.cpp = cpp_client
        self.model_fusion = model_fusion
        self.window = window

    def propose(self, descs: List[Dict[str,Any]]) -> List[Tuple[int,int]]:
        # descs: [{"tag": str, "qubits": List[int], "is_diag": bool, "is_fused": bool, "edge_nodes": int, ...}]
        # ここで Head B（融合モデル）に推論させ、[(start,end),...] を返す
        # 最初は安全ルール（対角の塊、短い1Q列、CX-RZ-CX）だけに限定するのが無難
        return []

    def run(self):
        while True:
            window_desc = self.cpp.snapshot_queue_window(self.window)  # list of dict
            if not window_desc:
                break
            ranges = self.propose(window_desc)
            if ranges:
                self.cpp.propose_fusion(ranges)
            cont = self.cpp.step()  # 1ステップ実行
            if not cont:
                break