// 自動生成: routing_n6 (6 量子ビット, 39 ゲート)
// 生成元: BQSim 同梱の MQT-Bench 1.1.0 回路 (IDEA-CUHK/BQSim circuits/routing_n6.qasm)
#include "routing6MQT.hpp"

void buildRouting6MQT(QuantumCircuit& qc) {
    qc.addRy(0, -1.6091521556564607);
    qc.addRy(1, -1.568191406613579);
    qc.addRy(2, -1.469942259085186);
    qc.addRy(3, 1.6246858840637424);
    qc.addRy(4, 1.687856548265116);
    qc.addRy(5, -1.5644608391463715);
    qc.addCX(4, 5);
    qc.addCX(3, 4);
    qc.addCX(2, 3);
    qc.addCX(1, 2);
    qc.addCX(0, 1);
    qc.addRy(0, -2.5384568520297575);
    qc.addRy(1, -1.5762010081061046);
    qc.addRy(2, -1.7151601595934756);
    qc.addRy(3, 1.2955637548649466);
    qc.addRy(4, 2.0692609481189055);
    qc.addRy(5, -0.0030759346664153073);
    qc.addCX(4, 5);
    qc.addCX(3, 4);
    qc.addCX(2, 3);
    qc.addCX(1, 2);
    qc.addCX(0, 1);
    qc.addRy(0, -0.4892811952799393);
    qc.addRy(1, -0.7537309450701157);
    qc.addRy(2, 1.5153866838329422);
    qc.addRy(3, -1.7229336573342435);
    qc.addRy(4, -1.0725300977013172);
    qc.addRy(5, 0.017670641518661795);
    qc.addCX(4, 5);
    qc.addCX(3, 4);
    qc.addCX(2, 3);
    qc.addCX(1, 2);
    qc.addCX(0, 1);
    qc.addRy(0, -2.049018501087617);
    qc.addRy(1, 0.41910807393610094);
    qc.addRy(2, 1.559432626994918);
    qc.addRy(3, 1.4194797279644502);
    qc.addRy(4, 1.3300726524968276);
    qc.addRy(5, 1.492714219983881);
}

// 既存の MQT 回路（qnn18MQT 等）と同じ自己完結形。main.cpp から単体で呼ぶ用。
// ベンチ掃引は BENCH_TABLE 経由で buildRouting6MQT を使う（GPU ビット数などを外から振るため）。
void routing6MQT() {
    array<int, 6> meas;
    QuantumCircuit qc(6);

    buildRouting6MQT(qc);

    meas[0] = qc.measure(0);
    meas[1] = qc.measure(1);
    meas[2] = qc.measure(2);
    meas[3] = qc.measure(3);
    meas[4] = qc.measure(4);
    meas[5] = qc.measure(5);

    return;
}
