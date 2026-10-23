#ifndef GRAPHSTATE20MQT_HPP
#define GRAPHSTATE20MQT_HPP

#include "../../models/state.hpp"
#include "../../common/mathUtils.hpp"
#include "../../models/circuit.hpp"
#include <array>

using namespace std;

void buildGraphState20MQT(QuantumCircuit& qc);   // 回路を組むだけ（BENCH_TABLE 用）
void graphState20MQT();         // 自己完結形（既存の MQT 回路と同じ）

#endif
