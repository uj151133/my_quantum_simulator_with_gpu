#include "qmddCore.hpp"


namespace {

// Ops -> QuantumCircuit の適用
static inline void apply_op(QuantumCircuit& qc, const Op& op) {
    const auto& t = op.gate_type;
    if (t == "I" || t == "ID") { /* no-op */ }
    else if (t == "H")   qc.addH(op.qubits.at(0));
    else if (t == "X")   qc.addX(op.qubits.at(0));
    else if (t == "Y")   qc.addY(op.qubits.at(0));
    else if (t == "Z")   qc.addZ(op.qubits.at(0));
    else if (t == "S")   qc.addS(op.qubits.at(0));
    else if (t == "SDG") qc.addSdg(op.qubits.at(0));
    else if (t == "T")   qc.addT(op.qubits.at(0));
    else if (t == "TDG") qc.addTdg(op.qubits.at(0));
    else if (t == "SX")  qc.addV(op.qubits.at(0)); // Qiskit sx -> addV
    else if (t == "RX")  qc.addRx(op.qubits.at(0), op.theta);
    else if (t == "RY")  qc.addRy(op.qubits.at(0), op.theta);
    else if (t == "RZ")  qc.addRz(op.qubits.at(0), op.theta);
    else if (t == "P")   qc.addP(op.qubits.at(0), op.theta);
    else if (t == "U1")  qc.addU1(op.qubits.at(0), op.theta);
    else if (t == "U2")  qc.addU2(op.qubits.at(0), op.phi, op.lam);
    else if (t == "U3")  qc.addU3(op.qubits.at(0), op.theta, op.phi, op.lam);
    else if (t == "CX" || t == "CNOT") qc.addCX(op.qubits.at(0), op.qubits.at(1));
    else if (t == "CY")  qc.addCY(op.qubits.at(0), op.qubits.at(1));
    else if (t == "CZ")  qc.addCZ(op.qubits.at(0), op.qubits.at(1));
    else if (t == "CP")  qc.addCP(op.qubits.at(0), op.qubits.at(1), op.theta); // thetaをphi扱い
    else if (t == "CH")  qc.addCH(op.qubits.at(0), op.qubits.at(1));
    else if (t == "CRX") qc.addCRx(op.qubits.at(0), op.qubits.at(1), op.theta);
    else if (t == "CRY") qc.addCRy(op.qubits.at(0), op.qubits.at(1), op.theta);
    else if (t == "CRZ") qc.addCRz(op.qubits.at(0), op.qubits.at(1), op.theta);
    else if (t == "CU")  qc.addCU(op.qubits.at(0), op.qubits.at(1), op.theta, op.phi, op.lam);
    else if (t == "SWAP")qc.addSWAP(op.qubits.at(0), op.qubits.at(1));
    else {
        // 未対応は無視（必要に応じて追加）
    }
}

} // anonymous namespace

// ===== Session =====
Session::Session(int num_qubits) : nq_(num_qubits) {}

py::dict Session::profile_chunk(const vector<Op>& ops) {
    QuantumCircuit qc(nq_);
    for (const auto& op : ops) apply_op(qc, op);

    auto nodes_before = UniqueTable::getInstance().getTotalEntryCount();
    auto t0 = chrono::high_resolution_clock::now();
    {
        // simulate中はGIL解放（Python側の他スレッドをブロックしない）
        py::gil_scoped_release release;
        qc.simulate();
    }
    auto t1 = chrono::high_resolution_clock::now();
    auto nodes_after = UniqueTable::getInstance().getTotalEntryCount();

    double wall_ms = chrono::duration<double, milli>(t1 - t0).count();
    long long delta = static_cast<long long>(nodes_after) - static_cast<long long>(nodes_before);
    if (delta < 0) delta = 0;

    py::dict out;
    out["wall_time_ms"] = wall_ms;
    out["nodes_peak"]   = static_cast<long long>(nodes_after);
    out["nodes_delta"]  = delta;
    return out;
}

// ===== pybind11 module =====
PYBIND11_MODULE(qmdd_core, m) {
    m.doc() = "QMDD simulator bindings (apply Ops and simulate)";

    py::class_<Op>(m, "Op")
        .def(py::init<>())
        .def_readwrite("gate_type", &Op::gate_type)
        .def_readwrite("qubits",    &Op::qubits)
        .def_readwrite("theta",     &Op::theta)
        .def_readwrite("phi",       &Op::phi)
        .def_readwrite("lam",       &Op::lam)
        .def_readwrite("is_diag",   &Op::is_diag);

    py::class_<Session>(m, "Session")
        .def(py::init<int>(), py::arg("num_qubits"))
        .def("profile_chunk", &Session::profile_chunk, py::arg("ops"));
}