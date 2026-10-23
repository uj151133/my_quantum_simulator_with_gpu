// 自動生成: graph_state_n20 (20 量子ビット, 40 ゲート)
// 生成元: BQSim 同梱の MQT-Bench 1.1.0 回路 (IDEA-CUHK/BQSim circuits/graph_state_n20.qasm)
#include "graphState20MQT.hpp"

void buildGraphState20MQT(QuantumCircuit& qc) {
    qc.addH(0);
    qc.addH(1);
    qc.addCZ(0, 1);
    qc.addH(2);
    qc.addH(3);
    qc.addCZ(2, 3);
    qc.addH(4);
    qc.addCZ(3, 4);
    qc.addH(5);
    qc.addH(6);
    qc.addCZ(5, 6);
    qc.addH(7);
    qc.addH(8);
    qc.addCZ(6, 8);
    qc.addCZ(7, 8);
    qc.addH(9);
    qc.addH(10);
    qc.addCZ(9, 10);
    qc.addH(11);
    qc.addCZ(9, 11);
    qc.addH(12);
    qc.addCZ(2, 12);
    qc.addH(13);
    qc.addCZ(10, 13);
    qc.addH(14);
    qc.addCZ(11, 14);
    qc.addH(15);
    qc.addCZ(0, 15);
    qc.addCZ(1, 15);
    qc.addH(16);
    qc.addCZ(4, 16);
    qc.addCZ(13, 16);
    qc.addH(17);
    qc.addCZ(7, 17);
    qc.addCZ(12, 17);
    qc.addH(18);
    qc.addCZ(14, 18);
    qc.addH(19);
    qc.addCZ(5, 19);
    qc.addCZ(18, 19);
}

// 既存の MQT 回路（qnn18MQT 等）と同じ自己完結形。main.cpp から単体で呼ぶ用。
// ベンチ掃引は BENCH_TABLE 経由で buildGraphState20MQT を使う（GPU ビット数などを外から振るため）。
void graphState20MQT() {
    array<int, 20> meas;
    QuantumCircuit qc(20);

    buildGraphState20MQT(qc);

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
    meas[16] = qc.measure(16);
    meas[17] = qc.measure(17);
    meas[18] = qc.measure(18);
    meas[19] = qc.measure(19);

    return;
}
