#include "bridgeModule.hpp"
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include "../opt/law.hpp"
#include "../models/circuit.hpp"

using namespace std;
namespace py = pybind11;

static inline void normalize_all(vector<Core>& v){
    for(auto& c: v) c.normalize();
}

vector<Core> optimize_ops(const vector<Core>& in_ops){
    vector<Core> ops = in_ops;
    normalize_all(ops);
    auto opt = law::optionsFromEnv(law::Options{});
    auto out = law::optimize(ops, opt);
    normalize_all(out);
    return out;
}

vector<Core> reorder_ops(const vector<Core>& ops, const vector<int>& order_indices){
    if(order_indices.size()!=ops.size()) throw runtime_error("reorder_ops: size mismatch");
    vector<Core> out(ops.size());
    for(size_t i=0;i<ops.size();++i){
        int idx = order_indices[i];
        if(idx<0 || (size_t)idx>=ops.size()) throw runtime_error("reorder_ops: index OOR");
        out[i] = ops[(size_t)idx];
    }
    return out;
}

vector<int> legalize_order_by_dag(const vector<Core>& ops, const vector<int>& perm){
    vector<Core> nops = ops;
    normalize_all(nops);
    return dag::tuneDAG(nops, perm);
}

static void emit_core(QuantumCircuit& qc, const Core& o){
    const string g = Core::upper(o.tag);
    if(g=="H") qc.addH(o.qubits.at(0));
    else if(g=="X") qc.addX(o.qubits.at(0));
    else if(g=="Y") qc.addY(o.qubits.at(0));
    else if(g=="Z") qc.addZ(o.qubits.at(0));
    else if(g=="P"||g=="U1") qc.addP(o.qubits.at(0), (o.phi!=0.0? o.phi : o.theta));
    else if(g=="RX") qc.addRx(o.qubits.at(0), o.theta);
    else if(g=="RY") qc.addRy(o.qubits.at(0), o.theta);
    else if(g=="RZ") qc.addRz(o.qubits.at(0), o.theta);
    else if(g=="CX"||g=="CNOT") qc.addCX(o.qubits.at(0), o.qubits.at(1));
    else if(g=="CZ") qc.addCZ(o.qubits.at(0), o.qubits.at(1));
    else if(g=="CP") qc.addCP(o.qubits.at(0), o.qubits.at(1), (o.phi!=0.0? o.phi : o.theta));
    else if(g=="CRZ") qc.addCRz(o.qubits.at(0), o.qubits.at(1), o.theta);
    else if(g=="SWAP") qc.addSWAP(o.qubits.at(0), o.qubits.at(1));
}

py::dict evaluate_runtime_for_ops(int nq, const vector<Core>& ops, const string&){
    vector<Core> nops = ops; normalize_all(nops);
    QuantumCircuit qc(nq);
    for(const auto& o: nops) emit_core(qc, o);
    auto t0 = chrono::high_resolution_clock::now();
    qc.simulate();
    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();
    py::dict d;
    d["wall_time_ms"] = ms;
    return d;
}

void register_qmdd_bridge(pybind11::module_& m){
    py::class_<Core>(m, "Core")
        .def(py::init<>())
        .def_readwrite("tag", &Core::tag)
        .def_readwrite("qubits", &Core::qubits)
        .def_readwrite("theta", &Core::theta)
        .def_readwrite("phi", &Core::phi)
        .def_readwrite("lam", &Core::lam)
        .def_readwrite("shape", &Core::shape)
        .def_readwrite("handle", &Core::handle)
        .def_readwrite("edge_nodes", &Core::edge_nodes);
    m.def("optimize_ops", &optimize_ops);
    m.def("reorder_ops", &reorder_ops);
    m.def("legalize_order_by_dag", &legalize_order_by_dag);
    m.def("evaluate_runtime_for_ops", &evaluate_runtime_for_ops);
}

PYBIND11_MODULE(qmdd_core, m){
    register_qmdd_bridge(m);
}