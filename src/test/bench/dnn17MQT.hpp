#ifndef DNN17MQT_HPP
#define DNN17MQT_HPP

#include "../../models/state.hpp"
#include "../../common/mathUtils.hpp"
#include "../../models/circuit.hpp"
#include <array>

using namespace std;

void buildDnn17MQT(QuantumCircuit& qc);   // 回路を組むだけ（BENCH_TABLE 用）
void dnn17MQT();                // 自己完結形（既存の MQT 回路と同じ）

#endif
