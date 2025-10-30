import qmdd_core as m

def test_qmdd_core_direct():
    sess = m.Session(3)

    ops = []
    op = m.Op(); op.gate_type = "H";  op.qubits = [0]; ops.append(op)
    op = m.Op(); op.gate_type = "CX"; op.qubits = [0, 1]; ops.append(op)
    op = m.Op(); op.gate_type = "RZ"; op.qubits = [1]; op.theta = 1.234; ops.append(op)

    res = sess.profile_chunk(ops)
    print("metrics:", dict(res))

if __name__ == "__main__":
    test_qmdd_core_direct()