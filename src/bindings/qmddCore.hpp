#ifndef QMDDCORE_HPP
#define QMDDCORE_HPP

#include <string>
#include <vector>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../models/circuit.hpp"
#include "../models/uniqueTable.hpp"
#include "../opt/law.hpp"

using namespace std;
namespace py = pybind11;

struct Op {
    string gate_type;
    vector<int> qubits;
    double theta = .0;
    double phi   = .0;
    double lam   = .0;
    int is_diag  = 0;              // 参照用
};

class Session {
public:
    explicit Session(int num_qubits);

    py::dict profile_chunk(const vector<Op>& ops);

private:
    int nq_{0};
};

#endif