#ifndef VQE14MQT_HPP
#define VQE14MQT_HPP

#include "../../models/state.hpp"
#include "../../common/mathUtils.hpp"
#include "../../models/circuit.hpp"
#include <array>

using namespace std;

void buildVqe14MQT(QuantumCircuit& qc);   // 回路を組むだけ（BENCH_TABLE 用）
void vqe14MQT();                // 自己完結形（既存の MQT 回路と同じ）

#endif
