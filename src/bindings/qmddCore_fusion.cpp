#include "qmddCore.hpp"
#include "../opt/law.hpp"
#include "../models/circuit.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;

// 既存の Session にアクセスできる体で（なければ別モジュールでも可）
PYBIND11_MODULE(qmdd_core, m) {
    // IR最適化のみ返す（前処理ループ用）
    m.def("optimize_ops", [](const std::vector<law::Op>& in_ops){
        law::Options opt = law::options_from_env(law::Options{});
        auto out = law::optimize(in_ops, opt);
        return out;
    });

    // GateDesc をPython辞書で返す（軽量）
    m.def("snapshot_queue_window", [](Session& s, int max_items){
        auto& qc = s.qc();
        auto v = qc.snapshotQueueWindow((size_t)std::max(0, max_items));
        py::list out;
        for (auto& d : v){
            py::dict o;
            o["tag"] = d.tag;
            o["qubits"] = d.qubits;
            o["is_diag"] = d.is_diag;
            o["is_fused"] = d.is_fused;
            o["handle"] = (unsigned long long)d.handle;
            o["edge_nodes"] = (unsigned long long)d.edge_nodes;
            out.append(o);
        }
        return out;
    });

    m.def("propose_fusion", [](Session& s, const std::vector<std::pair<int,int>>& ranges){
        s.qc().fuseRanges(ranges);
    });

    m.def("step", [](Session& s){
        return s.qc().simulateStep();
    });
}