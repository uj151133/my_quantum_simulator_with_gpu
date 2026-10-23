// 自動生成: vqe_n12 (12 量子ビット, 58 ゲート)
// 生成元: BQSim 同梱の MQT-Bench 1.1.0 回路 (IDEA-CUHK/BQSim circuits/vqe_n12.qasm)
#include "vqe12MQT.hpp"

void buildVqe12MQT(QuantumCircuit& qc) {
    qc.addRy(0, 3.141592653589793);
    qc.addRy(1, 9.42477796076938);
    qc.addRy(2, 9.578097716802171);
    qc.addRy(3, 9.534070592883062);
    qc.addRy(4, 9.544606710588393);
    qc.addRy(5, 2.95517996809816);
    qc.addRy(6, 0.173614704874243);
    qc.addRy(7, 12.564994546716733);
    qc.addRy(8, 11.010656082220622);
    qc.addRy(9, 3.13390036081138);
    qc.addRy(10, 11.003823141596524);
    qc.addRy(11, 10.545078204758582);
    qc.addCX(10, 11);
    qc.addCX(9, 10);
    qc.addRy(11, 0.0040136750135147);
    qc.addCX(8, 9);
    qc.addRy(10, 3.13159824181319);
    qc.addCX(7, 8);
    qc.addRy(9, 2.11911274357807);
    qc.addCX(10, 11);
    qc.addCX(6, 7);
    qc.addRy(8, 11.544189246001674);
    qc.addCX(9, 10);
    qc.addRy(11, 11.445444133616482);
    qc.addCX(5, 6);
    qc.addRy(7, 10.997509545019023);
    qc.addCX(8, 9);
    qc.addRy(10, 10.995547557213243);
    qc.addCX(4, 5);
    qc.addRy(6, 1.57046398139895);
    qc.addCX(7, 8);
    qc.addRy(9, 0.0177510928738939);
    qc.addCX(3, 4);
    qc.addRy(5, 1.5693447644873002);
    qc.addCX(6, 7);
    qc.addRy(8, 10.994561170787271);
    qc.addCX(2, 3);
    qc.addRy(4, 11.016454130033534);
    qc.addCX(5, 6);
    qc.addRy(7, 1.74353834232086);
    qc.addCX(1, 2);
    qc.addRy(3, 10.806930007387923);
    qc.addCX(4, 5);
    qc.addRy(6, 1.75880793356509);
    qc.addCX(0, 1);
    qc.addRy(2, 11.889929092617818);
    qc.addCX(3, 4);
    qc.addRy(5, 1.69086111455446);
    qc.addRy(0, 12.564737367831654);
    qc.addRy(1, 3.141592653589793);
    qc.addCX(2, 3);
    qc.addRy(4, 1.68189586281483);
    qc.addCX(1, 2);
    qc.addRy(3, 1.32974612369579);
    qc.addCX(0, 1);
    qc.addRy(2, 11.87612456103887);
    qc.addRy(0, 12.566338410816225);
    qc.addRy(1, 0.00126561347749662);
}

// 既存の MQT 回路（qnn18MQT 等）と同じ自己完結形。main.cpp から単体で呼ぶ用。
// ベンチ掃引は BENCH_TABLE 経由で buildVqe12MQT を使う（GPU ビット数などを外から振るため）。
void vqe12MQT() {
    array<int, 12> meas;
    QuantumCircuit qc(12);

    buildVqe12MQT(qc);

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

    return;
}
