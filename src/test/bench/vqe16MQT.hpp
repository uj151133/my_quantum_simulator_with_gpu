#ifndef VQE16MQT_HPP
#define VQE16MQT_HPP

#include "../../models/state.hpp"
#include "../../common/mathUtils.hpp"
#include "../../models/circuit.hpp"
#include <array>

using namespace std;

void buildVqe16MQT(QuantumCircuit& qc);   // 回路を組むだけ（BENCH_TABLE 用）
void vqe16MQT();                // 自己完結形（既存の MQT 回路と同じ）

#endif
