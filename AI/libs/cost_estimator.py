from .signatures import Signature, simulate_update, signature_cost

class HeuristicCostEstimator:
    def __init__(self, cfg):
        self.cfg = cfg

    def delta_cost(self, prefix: Signature, gate_sig: Signature) -> float:
        before = signature_cost(prefix, self.cfg)
        after = signature_cost(simulate_update(prefix, gate_sig, self.cfg.top_k_levels), self.cfg)
        return after - before