// 自動生成: graph_state_n16 (16 量子ビット, 32 ゲート)
// 生成元: BQSim 同梱の MQT-Bench 1.1.0 回路 (IDEA-CUHK/BQSim circuits/graph_state_n16.qasm)
#include "graphState16MQT.hpp"

void buildGraphState16MQT(QuantumCircuit& qc) {
    qc.addH(0);
    qc.addH(1);
    qc.addCZ(0, 1);
    qc.addH(2);
    qc.addH(3);
    qc.addCZ(2, 3);
    qc.addH(4);
    qc.addH(5);
    qc.addCZ(4, 5);
    qc.addH(6);
    qc.addCZ(2, 6);
    qc.addH(7);
    qc.addCZ(0, 7);
    qc.addH(8);
    qc.addH(9);
    qc.addCZ(3, 9);
    qc.addCZ(8, 9);
    qc.addH(10);
    qc.addH(11);
    qc.addCZ(1, 11);
    qc.addCZ(10, 11);
    qc.addH(12);
    qc.addCZ(10, 12);
    qc.addH(13);
    qc.addCZ(5, 13);
    qc.addCZ(7, 13);
    qc.addH(14);
    qc.addCZ(4, 14);
    qc.addCZ(8, 14);
    qc.addH(15);
    qc.addCZ(6, 15);
    qc.addCZ(12, 15);
}

// 既存の MQT 回路（qnn18MQT 等）と同じ自己完結形。main.cpp から単体で呼ぶ用。
// ベンチ掃引は BENCH_TABLE 経由で buildGraphState16MQT を使う（GPU ビット数などを外から振るため）。
void graphState16MQT() {
    array<int, 16> meas;
    QuantumCircuit qc(16);

    buildGraphState16MQT(qc);

    meas[0] = qc.measure(0);
    meas[1] = qc.measure(1);
    meas[2] = qc.measure(2);
    meas[3] = qc.measure(3);
    meas[4] = qc.measure(4);
    meas[5] = qc.measure(5);
    meas[6] = qc.measure(6);
    meas[7] = qc.measure(7);
    meas[8] = qc.measure(8);
    meas[9] = qc.measure(9);
    meas[10] = qc.measure(10);
    meas[11] = qc.measure(11);
    meas[12] = qc.measure(12);
    meas[13] = qc.measure(13);
    meas[14] = qc.measure(14);
    meas[15] = qc.measure(15);

    return;
}
