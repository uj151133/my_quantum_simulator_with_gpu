#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../opt/law.hpp"
#include "../common/Core.hpp"
#include "../models/circuit.hpp"

namespace py = pybind11;

using namespace std;

PYBIND11_MODULE(qmdd_bridge, m) {
    m.def("optimize_ops", [](const std::vector<Core>& in_ops){
        law::Options opt = law::options_from_env(law::Options{});
        return law::optimize(in_ops, opt);
    });
    m.def("snapshot_queue_window", [](Session& s, int max_items){
        return s.qc().snapshotQueueWindow((size_t)std::max(0, max_items));
    });
    m.def("propose_fusion", [](Session& s, const std::vector<std::pair<int,int>>& ranges){
        s.qc().fuseRanges(ranges);
    });
    m.def("step", [](Session& s){
        return s.qc().simulateStep();
    });
}