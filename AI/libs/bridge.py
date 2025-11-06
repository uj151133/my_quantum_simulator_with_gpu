# ...existing code...
class Bridge:
    def __init__(self):
        try:
            from AI import qmdd_core as _b
            self._b = _b
            self._err = None
        except Exception as e1:
            try:
                import qmdd_core as _b
                self._b = _b
                self._err = None
            except Exception as e2:
                self._b = None
                self._err = (e1, e2)
        # Core型の参照を保持
        self._CoreCls = getattr(self._b, "Core", None) if self._b else None
        self._diag_tags = {"RZ","U1","P","S","T","Z","CZ","CP","CRZ","RZZ"}

    def _check(self):
        if self._b is None:
            raise RuntimeError(f"qmdd_core is not available: {self._err}")

    def _is_core(self, obj) -> bool:
        # return obj.__class__.__name__ == "Core"
        return (self._CoreCls is not None) and isinstance(obj, self._CoreCls)

    def _to_core_list(self, ops):
        # if not ops:
        #     return ops
        # if self._is_core(ops[0]):
        #     return ops
        # C = self._CoreCls
        # if C is None:
        #     return ops
        # out = []
        # for d in ops:
        #     c = C()
        #     c.tag = d.get("tag","")
        #     c.qubits = d.get("qubits",[])
        #     c.shape = d.get("shape","")
        #     c.theta = float(d.get("theta",0.0))
        #     c.phi = float(d.get("phi",0.0))
        #     c.lam = float(d.get("lam",0.0))
        #     out.append(c)
        # return out
        if not ops:
            return []
        # 既に C++ Core の配列ならそのまま
        if self._is_core(ops[0]):
            return ops
        C = self._CoreCls
        if C is None:
            return ops
        out = []
        for o in ops:
            c = C()
            if isinstance(o, dict):
                c.tag = o.get("tag", "")
                c.qubits = o.get("qubits", [])
                c.shape = o.get("shape", "")
                c.theta = float(o.get("theta", 0.0))
                c.phi = float(o.get("phi", 0.0))
                c.lam = float(o.get("lam", 0.0))
            else:
                # Python側 Core(dataclass) からコピー
                c.tag = getattr(o, "tag", "")
                c.qubits = list(getattr(o, "qubits", []))
                c.shape = getattr(o, "shape", "")
                c.theta = float(getattr(o, "theta", 0.0))
                c.phi = float(getattr(o, "phi", 0.0))
                c.lam = float(getattr(o, "lam", 0.0))
            out.append(c)
        return out

    def _core_to_dict(self, c):
        # C++ Core → dict
        tag = str(getattr(c, "tag", "")).upper()
        qubits = list(getattr(c, "qubits", []))
        shape = getattr(c, "shape", "")
        shape = str(shape) if isinstance(shape, str) and shape else ("DIAG" if tag in self._diag_tags else "GENERAL")
        return {
            "tag": tag,
            "qubits": qubits,
            "shape": shape,
            "theta": float(getattr(c, "theta", 0.0)),
            "phi": float(getattr(c, "phi", 0.0)),
            "lam": float(getattr(c, "lam", 0.0)),
        }

    def legalize_order_by_dag(self, ops, perm):
        self._check()
        return self._b.legalize_order_by_dag(self._to_core_list(ops), list(map(int, perm)))

    def evaluate_runtime_for_ops(self, nq: int, ops, name: str):
        self._check()
        return self._b.evaluate_runtime_for_ops(int(nq), self._to_core_list(ops), str(name))

    # ここを必ず dict で返す
    def optimize_ops(self, ops):
        self._check()
        res = self._b.optimize_ops(self._to_core_list(ops))
        if isinstance(res, list) and res and self._is_core(res[0]):
            return [self._core_to_dict(x) for x in res]
        return res
