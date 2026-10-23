// 自動生成: routing_n12 (12 量子ビット, 81 ゲート)
// 生成元: BQSim 同梱の MQT-Bench 1.1.0 回路 (IDEA-CUHK/BQSim circuits/routing_n12.qasm)
#include "routing12MQT.hpp"

void buildRouting12MQT(QuantumCircuit& qc) {
    qc.addRy(0, -0.0512083984166585);
    qc.addRy(1, 0.07324755524157076);
    qc.addRy(2, -2.5176269171846557);
    qc.addRy(3, -0.010299937850382957);
    qc.addRy(4, 0.040427641943305406);
    qc.addRy(5, 1.610738605722133);
    qc.addRy(6, 0.016834720784235355);
    qc.addRy(7, 1.5222625428198393);
    qc.addRy(8, 0.07121920835658857);
    qc.addRy(9, -0.5183727684691843);
    qc.addRy(10, -0.511289262522007);
    qc.addRy(11, 0.6467551194950519);
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
    qc.addRy(0, -1.8786363117976832);
    qc.addRy(1, 2.486114234590843);
    qc.addRy(2, -1.468948228528621);
    qc.addRy(3, -1.331622032639472);
    qc.addRy(4, -0.960801190341064);
    qc.addRy(5, 1.2948956932774736);
    qc.addRy(6, -0.21080955404477833);
    qc.addRy(7, 0.41754047028053265);
    qc.addRy(8, 0.346232737971357);
    qc.addRy(9, 1.3845711988658318);
    qc.addRy(10, -1.2280195964526384);
    qc.addRy(11, -2.292912727392617);
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
    qc.addRy(0, 1.0892414769461425);
    qc.addRy(1, 1.1650466711809235);
    qc.addRy(2, 1.617306598304894);
    qc.addRy(3, -1.516176711571421);
    qc.addRy(4, 1.5733885331242798);
    qc.addRy(5, 1.6168857856396381);
    qc.addRy(6, 0.12633802764072832);
    qc.addRy(7, -0.05967767556712726);
    qc.addRy(8, -1.6184432606845907);
    qc.addRy(9, -4.451965782491241e-06);
    qc.addRy(10, 0.501531292745206);
    qc.addRy(11, 0.3578083948992959);
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
    qc.addRy(0, -1.7334335024900418);
    qc.addRy(1, -0.24307073231969123);
    qc.addRy(2, -2.063936872842193);
    qc.addRy(3, -0.08882828629899524);
    qc.addRy(4, -2.863459272028703);
    qc.addRy(5, 2.488130119965415);
    qc.addRy(6, -0.007570883042990317);
    qc.addRy(7, -1.5534931194197028);
    qc.addRy(8, -1.5509482467380937);
    qc.addRy(9, 1.4969137182062546);
    qc.addRy(10, -1.5898306456679157);
    qc.addRy(11, 1.567170008247618);
}

// 既存の MQT 回路（qnn18MQT 等）と同じ自己完結形。main.cpp から単体で呼ぶ用。
// ベンチ掃引は BENCH_TABLE 経由で buildRouting12MQT を使う（GPU ビット数などを外から振るため）。
void routing12MQT() {
    array<int, 12> meas;
    QuantumCircuit qc(12);

    buildRouting12MQT(qc);

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
