#include <iostream>
#include <cstdlib>
#include <cstdlib>
#include <string>
#include <unistd.h>
// #include <ginac/ginac.h>

#include "src/models/bit.hpp"
#include "src/models/gate.hpp"
#include "src/models/circuit.hpp"
#include "src/models/uniqueTable.hpp"
#include "src/models/qmdd.hpp"
#include "src/common/mathUtils.hpp"
#include "src/common/calculation.hpp"
#include "src/common/monitor.hpp"

// using namespace GiNaC;
using namespace std;

void execute() {
    UniqueTable& uniqueTable = UniqueTable::getInstance();
    // QMDDEdge firstEdge = mathUtils::kroneckerProduct(state::KET_0().getInitialEdge(), state::KET_0().getInitialEdge());
    // QuantumCircuit circuit(2, QMDDState(firstEdge));
    // circuit.addI(0);
    // circuit.addI(1);
    // circuit.execute();/

    // QMDDGate zeroGate = gate::O();

    QMDDGate i1Gate = gate::I();
    QMDDGate i2Gate = gate::I();
    // QMDDGate phGate = gate::Ph(0.5);
    QMDDGate xGate = gate::X();
    // QMDDGate hGate = gate::H();
    // QMDDGate sGate = gate::S();
    // QMDDGate toffGate = gate::Toff();
    // QMDDGate ffredkinGate = gate::fFredkin();
    // QMDDState ket0State = state::KET_0();
    // QMDDState ket1State = state::KET_1();
    // QMDDState bra0State = state::BRA_0();
    // cout << "zeroGate:" << zeroGate.getInitialEdge() << endl;
    // cout << "zeroGate:" << zeroGate << endl;
    // cout << "igate:" << iGate.getInitialEdge() << endl;
    // cout << "phgate:" << phGate.getInitialEdge() << endl;
    // cout << "cx1gate:" << cx2Gate.getInitialEdge() << endl;
    // cout << "cx2gate:" << cx2Gate.getDepth() << endl;
    // cout << "igate:" << gate::I().getInitialEdge() << endl;
    // cout << "x1gate:" << gate::X().getInitialEdge() << endl;
    // QMDDGate h2Gate = gate::H();
    // cout << "hgate:" << hGate.getInitialEdge() << endl;
    // cout << "ket0" << ket0State.getInitialEdge().uniqueTableKey << endl;
    // cout << "bra0" << bra0State.getInitialEdge().uniqueTableKey << endl;
    // cout << "ket0:" << ket0State.getStartNode()->edges.size() << ", " << ket0State.getStartNode()->edges[0].size() << endl;
    // cout << "bra0:" << bra0State.getStartNode()->edges.size() << ", " << bra0State.getStartNode()->edges[0].size() << endl;
    // QMDDGate xGate = gate::X();
    // cout << "xgate:" << xGate.getInitialEdge() << endl;
    // QMDDState ket0 = state::KET_0();
    auto result1 = mathUtils::kroneckerProduct(i1Gate.getInitialEdge(), i2Gate.getInitialEdge());
    cout << "result1:" << result1 << endl;
    auto result2 = mathUtils::addition(xGate.getInitialEdge(), i1Gate.getInitialEdge());
    cout << "result2:" << result2 << endl;

    uniqueTable.printAllEntries();


    // QMDDGate cx1 = gate::CX1();
    // QMDDGate cx2 = gate::CX2();
    // auto result2 = mathUtils::addition(cx1.getInitialEdge(), cx2.getInitialEdge());
}

int main() {
    string processType = getProcessType();
    if (processType == "sequential") {
        cout << "逐次処理を実行します。" << endl;
        sequentialProcessing();
    } else if (processType == "multi-thread") {
        cout << "マルチスレッド処理を実行します。" << endl;
        parallelProcessing();
    } else if (processType == "multi-fiber") {
        cout << "マルチファイバー処理を実行します。" << endl;
        fiberProcessing();
    } else {
        cerr << "不明な処理タイプ: " << processType << endl;
    }
    printMemoryUsage();
    printMemoryUsageOnMac();

    bool isGuiEnabled = isExecuteGui();

    if (isGuiEnabled) {
        cout << "GUI is enabled." << endl;
    } else {
        cout << "GUI is disabled." << endl;
    }

    measureExecutionTime(execute);

    printMemoryUsage();
    printMemoryUsageOnMac();
    return 0;
}

