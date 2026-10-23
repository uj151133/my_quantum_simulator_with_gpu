// 自動生成: vqe_n14 (14 量子ビット, 68 ゲート)
// 生成元: BQSim 同梱の MQT-Bench 1.1.0 回路 (IDEA-CUHK/BQSim circuits/vqe_n14.qasm)
#include "vqe14MQT.hpp"

void buildVqe14MQT(QuantumCircuit& qc) {
    qc.addRy(0, 0.0009961745458614308);
    qc.addRy(1, 2.7989288521833586);
    qc.addRy(2, 0.02009177936139419);
    qc.addRy(3, -3.0623768856364095);
    qc.addRy(4, 0.3924106428692202);
    qc.addRy(5, -3.010342348051929);
    qc.addRy(6, 5.921800769174599e-05);
    qc.addRy(7, 1.5710967267103366);
    qc.addRy(8, -3.141592653589793);
    qc.addRy(9, 1.5707567678302206);
    qc.addRy(10, 0.00016770613619868687);
    qc.addRy(11, -2.0002951388018473);
    qc.addRy(12, 1.939527128126391);
    qc.addRy(13, 1.4691992189778686e-05);
    qc.addCX(12, 13);
    qc.addCX(11, 12);
    qc.addCX(10, 11);
    qc.addCX(9, 10);
    qc.addCX(8, 9);
    qc.addCX(7, 8);
    qc.addCX(6, 7);
    qc.addCX(5, 6);
    qc.addCX(4, 5);
    qc.addCX(3, 4);
    qc.addCX(2, 3);
    qc.addCX(1, 2);
    qc.addCX(0, 1);
    qc.addRy(0, -0.0014361858186602151);
    qc.addRy(1, -2.2826600979537512);
    qc.addRy(2, 1.288709249059189);
    qc.addRy(3, -1.5761481273849975);
    qc.addRy(4, 1.5712361398484467);
    qc.addRy(5, -1.5706588464312763);
    qc.addRy(6, -1.5708163781293154);
    qc.addRy(7, 2.0240933838096193);
    qc.addRy(8, -2.024063008743955);
    qc.addRy(9, 2.415281958735069);
    qc.addRy(10, 0.7262933702505351);
    qc.addRy(11, -1.5707224527750592);
    qc.addRy(12, 1.5708186271218882);
    qc.addRy(13, -1.5707318645266846);
    qc.addCX(12, 13);
    qc.addCX(11, 12);
    qc.addCX(10, 11);
    qc.addCX(9, 10);
    qc.addCX(8, 9);
    qc.addCX(7, 8);
    qc.addCX(6, 7);
    qc.addCX(5, 6);
    qc.addCX(4, 5);
    qc.addCX(3, 4);
    qc.addCX(2, 3);
    qc.addCX(1, 2);
    qc.addCX(0, 1);
    qc.addRy(0, -0.0018681207141472663);
    qc.addRy(1, 0.9079281741326347);
    qc.addRy(2, -1.130600363864126);
    qc.addRy(3, -1.5916871884987087);
    qc.addRy(4, -1.491529389832553);
    qc.addRy(5, -1.178417663649344);
    qc.addRy(6, 1.7021200798704101);
    qc.addRy(7, -1.5706444168470408);
    qc.addRy(8, -3.141592653589793);
    qc.addRy(9, -1.5707010044575682);
    qc.addRy(10, 3.141592653589793);
    qc.addRy(11, 1.570811140765109);
    qc.addRy(12, 2.712132887596124);
    qc.addRy(13, -2.772705485488589);
}

// 既存の MQT 回路（qnn18MQT 等）と同じ自己完結形。main.cpp から単体で呼ぶ用。
// ベンチ掃引は BENCH_TABLE 経由で buildVqe14MQT を使う（GPU ビット数などを外から振るため）。
void vqe14MQT() {
    array<int, 14> meas;
    QuantumCircuit qc(14);

    buildVqe14MQT(qc);

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

    return;
}
