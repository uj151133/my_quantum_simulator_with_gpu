// 自動生成: vqe_n16 (16 量子ビット, 78 ゲート)
// 生成元: BQSim 同梱の MQT-Bench 1.1.0 回路 (IDEA-CUHK/BQSim circuits/vqe_n16.qasm)
#include "vqe16MQT.hpp"

void buildVqe16MQT(QuantumCircuit& qc) {
    qc.addRy(0, 3.13973894554034);
    qc.addRy(1, 1.4925136011457);
    qc.addRy(2, -8.9920665716782e-05);
    qc.addRy(3, -1.23257522218457);
    qc.addRy(4, 1.86054637652338);
    qc.addRy(5, 2.61201045851137);
    qc.addRy(6, 0.319115428371679);
    qc.addRy(7, -1.47022955753925);
    qc.addRy(8, -2.12360657717647);
    qc.addRy(9, 2.37397708003484);
    qc.addRy(10, -2.07913035042815);
    qc.addRy(11, -2.92960134875116);
    qc.addRy(12, 2.83093527610263);
    qc.addRy(13, -0.0127668597893102);
    qc.addRy(14, -1.57124469944394);
    qc.addRy(15, 3.141592653589793);
    qc.addCX(14, 15);
    qc.addCX(13, 14);
    qc.addCX(12, 13);
    qc.addCX(11, 12);
    qc.addCX(10, 11);
    qc.addRy(11, 1.57079329385953);
    qc.addRy(12, 1.57076320854782);
    qc.addRy(13, -1.57081961034328);
    qc.addRy(14, 1.47379781943072);
    qc.addRy(15, 1.47389144321857);
    qc.addCX(14, 15);
    qc.addCX(13, 14);
    qc.addCX(12, 13);
    qc.addCX(11, 12);
    qc.addRy(12, 1.78274501735686);
    qc.addRy(13, -1.26033905435792);
    qc.addRy(14, 1.55796048690872);
    qc.addRy(15, 3.141592653589793);
    qc.addCX(9, 10);
    qc.addRy(10, 1.57083940178388);
    qc.addCX(10, 11);
    qc.addRy(11, -0.508317471527251);
    qc.addCX(8, 9);
    qc.addCX(7, 8);
    qc.addCX(6, 7);
    qc.addCX(5, 6);
    qc.addCX(4, 5);
    qc.addCX(3, 4);
    qc.addCX(2, 3);
    qc.addCX(1, 2);
    qc.addCX(0, 1);
    qc.addRy(0, 3.08922570257331);
    qc.addRy(1, -0.248645933369785);
    qc.addRy(2, 0.249379145067341);
    qc.addRy(3, -1.57092146383541);
    qc.addRy(4, -1.5707704269882);
    qc.addRy(5, -1.57073354865701);
    qc.addRy(6, -1.57081347984267);
    qc.addRy(7, 1.57076694138358);
    qc.addRy(8, -1.57079002828038);
    qc.addRy(9, 1.57094441646279);
    qc.addCX(9, 10);
    qc.addRy(10, -2.3384683897403);
    qc.addCX(8, 9);
    qc.addCX(7, 8);
    qc.addCX(6, 7);
    qc.addCX(5, 6);
    qc.addCX(4, 5);
    qc.addCX(3, 4);
    qc.addCX(2, 3);
    qc.addCX(1, 2);
    qc.addCX(0, 1);
    qc.addRy(0, -0.0524172519797756);
    qc.addRy(1, 1.64677588035859);
    qc.addRy(2, -0.0193121701337888);
    qc.addRy(3, -0.332992055921446);
    qc.addRy(4, 0.338261348290758);
    qc.addRy(5, -0.289668053994937);
    qc.addRy(6, -1.16523586650075);
    qc.addRy(7, -1.25167089132833);
    qc.addRy(8, -3.04098757877868);
    qc.addRy(9, 0.552777507970122);
}

// 既存の MQT 回路（qnn18MQT 等）と同じ自己完結形。main.cpp から単体で呼ぶ用。
// ベンチ掃引は BENCH_TABLE 経由で buildVqe16MQT を使う（GPU ビット数などを外から振るため）。
void vqe16MQT() {
    array<int, 16> meas;
    QuantumCircuit qc(16);

    buildVqe16MQT(qc);

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
