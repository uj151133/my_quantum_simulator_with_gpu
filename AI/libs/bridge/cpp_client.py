import importlib

class CppClient:
    def __init__(self, num_qubits: int):
        self.core = importlib.import_module("qmdd_core")
        self.sess = self.core.Session(num_qubits)

    def evaluate_chunk_runtime(self, circuit_id: str, ops):
        # ops: list of dicts produced by circuit_to_ops
        native_ops = []
        for op in ops:
            o = self.core.Op()
            o.gate_type = op.get("gate_type", "")
            o.qubits    = op.get("qubits", [])
            o.theta     = float(op.get("theta", 0.0))
            o.phi       = float(op.get("phi",   0.0))
            o.lam       = float(op.get("lam",   0.0))
            o.is_diag   = int(op.get("is_diag", 0))
            native_ops.append(o)
        res = self.sess.profile_chunk(native_ops)
        return {"wall_time_ms": float(res["wall_time_ms"]),
                "nodes_peak":   int(res["nodes_peak"]),
                "nodes_delta":  int(res["nodes_delta"])}