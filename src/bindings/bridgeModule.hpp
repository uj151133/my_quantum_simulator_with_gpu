#ifndef BRIDGEMODULE_HPP
#define BRIDGEMODULE_HPP
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../common/Core.hpp"
#include "../models/dag.hpp"
#include "../opt/law.hpp"
#include "../models/circuit.hpp"

using namespace std;

// law（C++）: 本物の最適化（env付きオプションで law::optimize を呼ぶ）
vector<Core> optimize_ops(const vector<Core>& in_ops);

// IR 並べ替え（ops -> ops）
vector<Core> reorder_ops(const vector<Core>& ops, const vector<int>& order_indices);

// DAG合法化（ops, perm -> order）
vector<int> legalize_order_by_dag(const vector<Core>& ops, const vector<int>& perm);

// 単発計測（実シミュレータで wall-clock を測定）
pybind11::dict evaluate_runtime_for_ops(int nq, const vector<Core>& ops, const string& name);

// pybind 登録
void register_qmdd_bridge(pybind11::module_& m);

#endif