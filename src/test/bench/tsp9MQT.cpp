// 自動生成: tsp_n9 (9 量子ビット, 94 ゲート)
// 生成元: BQSim 同梱の MQT-Bench 1.1.0 回路 (IDEA-CUHK/BQSim circuits/tsp_n9.qasm)
#include "tsp9MQT.hpp"

void buildTsp9MQT(QuantumCircuit& qc) {
    qc.addRy(0, 1.34607978454793);
    qc.addRy(1, 4.03501491300976);
    qc.addRy(2, 11.399365644399303);
    qc.addRy(3, 6.05280474556801);
    qc.addRy(4, 3.08551543256946);
    qc.addRy(5, 11.315959741934213);
    qc.addRy(6, 0.660094467150742);
    qc.addRy(7, 3.04317813769365);
    qc.addRy(8, 4.49473592347063);
    qc.addCZ(0, 1);
    qc.addRy(0, 6.815943763008739);
    qc.addCZ(1, 2);
    qc.addRy(1, 8.235906015181863);
    qc.addCZ(2, 3);
    qc.addCZ(0, 1);
    qc.addRy(2, 3.65712126117725);
    qc.addCZ(3, 4);
    qc.addRy(0, 11.761800752739324);
    qc.addCZ(1, 2);
    qc.addRy(3, 8.204714575416812);
    qc.addCZ(4, 5);
    qc.addRy(1, 8.770300707906802);
    qc.addCZ(2, 3);
    qc.addRy(4, 3.29262994777663);
    qc.addCZ(5, 6);
    qc.addCZ(0, 1);
    qc.addRy(2, 10.915681075831053);
    qc.addCZ(3, 4);
    qc.addRy(5, 9.983865452504173);
    qc.addCZ(6, 7);
    qc.addRy(0, 1.30828665301312);
    qc.addCZ(1, 2);
    qc.addRy(3, 1.00689096569379);
    qc.addCZ(4, 5);
    qc.addRy(6, 4.59258865718568);
    qc.addCZ(7, 8);
    qc.addRy(1, 9.543008094994583);
    qc.addCZ(2, 3);
    qc.addRy(4, 10.637117034670062);
    qc.addCZ(5, 6);
    qc.addRy(7, 10.199258136208421);
    qc.addRy(8, 4.44988195540909);
    qc.addCZ(0, 1);
    qc.addRy(2, 8.144931421842843);
    qc.addCZ(3, 4);
    qc.addRy(5, 2.43966146647018);
    qc.addCZ(6, 7);
    qc.addRy(0, 9.682444411636162);
    qc.addCZ(1, 2);
    qc.addRy(3, 6.8022232675723915);
    qc.addCZ(4, 5);
    qc.addRy(6, 8.569722547416752);
    qc.addCZ(7, 8);
    qc.addRy(1, 8.191312816327352);
    qc.addCZ(2, 3);
    qc.addRy(4, 9.668792209246652);
    qc.addCZ(5, 6);
    qc.addRy(7, 9.793906776190212);
    qc.addRy(8, 5.21162716660689);
    qc.addCZ(0, 1);
    qc.addRy(2, 0.51606877721084);
    qc.addCZ(3, 4);
    qc.addRy(5, 10.405789955451484);
    qc.addCZ(6, 7);
    qc.addRy(0, 3.8881027030372506);
    qc.addCZ(1, 2);
    qc.addRy(3, 11.622494887594437);
    qc.addCZ(4, 5);
    qc.addRy(6, 1.59006862234717);
    qc.addCZ(7, 8);
    qc.addRy(1, 5.6644704807054);
    qc.addCZ(2, 3);
    qc.addRy(4, 0.262807481241301);
    qc.addCZ(5, 6);
    qc.addRy(7, 5.09683961507163);
    qc.addRy(8, 1.56860661544026);
    qc.addRy(2, 5.04930243605052);
    qc.addCZ(3, 4);
    qc.addRy(5, 11.723328423938613);
    qc.addCZ(6, 7);
    qc.addRy(3, 4.15066432391157);
    qc.addCZ(4, 5);
    qc.addRy(6, 0.751783996334171);
    qc.addCZ(7, 8);
    qc.addRy(4, 3.82547521554831);
    qc.addCZ(5, 6);
    qc.addRy(7, 8.311091670844442);
    qc.addRy(8, 9.711520892262952);
    qc.addRy(5, 11.687029278123275);
    qc.addCZ(6, 7);
    qc.addRy(6, 2.11077635324514);
    qc.addCZ(7, 8);
    qc.addRy(7, 1.65800563819574);
    qc.addRy(8, 11.564327614215433);
}

// 既存の MQT 回路（qnn18MQT 等）と同じ自己完結形。main.cpp から単体で呼ぶ用。
// ベンチ掃引は BENCH_TABLE 経由で buildTsp9MQT を使う（GPU ビット数などを外から振るため）。
void tsp9MQT() {
    array<int, 9> meas;
    QuantumCircuit qc(9);

    buildTsp9MQT(qc);

    meas[0] = qc.measure(0);
    meas[1] = qc.measure(1);
    meas[2] = qc.measure(2);
    meas[3] = qc.measure(3);
    meas[4] = qc.measure(4);
    meas[5] = qc.measure(5);
    meas[6] = qc.measure(6);
    meas[7] = qc.measure(7);
    meas[8] = qc.measure(8);

    return;
}
