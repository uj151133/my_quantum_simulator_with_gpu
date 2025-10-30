from __future__ import annotations
from typing import Iterable
from qiskit import QuantumCircuit
import random, math

def gen_random_1q_depth(nq=8, depth=64, seed=None) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(nq)
    oneq = ["x","h","rz","t","s","sdg","tdg","sx"]
    for _ in range(depth):
        q = rng.randrange(nq)
        g = rng.choice(oneq)
        if g == "rz":
            qc.rz(2*math.pi*rng.random(), q)
        elif g == "x": qc.x(q)
        elif g == "h": qc.h(q)
        elif g == "t": qc.t(q)
        elif g == "s": qc.s(q)
        elif g == "sdg": qc.sdg(q)
        elif g == "tdg": qc.tdg(q)
        elif g == "sx": qc.sx(q)
    return qc

def gen_random_2q_cz(nq=8, depth=64, seed=None) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(nq)
    for _ in range(depth):
        a = rng.randrange(nq); b = rng.randrange(nq)
        if a == b: b = (b+1) % nq
        qc.h(a); qc.cz(a,b); qc.h(a)
    return qc

def dataset_stream(kind="mix", limit=None, seed=None) -> Iterable[QuantumCircuit]:
    rng = random.Random(seed)
    i = 0
    while limit is None or i < limit:
        if kind == "1q":
            yield gen_random_1q_depth(nq=8, depth=128, seed=rng.randrange(1<<30))
        elif kind == "2q":
            yield gen_random_2q_cz(nq=8, depth=128, seed=rng.randrange(1<<30))
        else:
            if rng.random() < 0.5:
                yield gen_random_1q_depth(nq=8, depth=128, seed=rng.randrange(1<<30))
            else:
                yield gen_random_2q_cz(nq=8, depth=128, seed=rng.randrange(1<<30))
        i += 1