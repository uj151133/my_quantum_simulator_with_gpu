#include <iostream>
#include <cstdlib>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include "src/models/state.hpp"
#include "src/models/gate.hpp"
#include "src/models/circuit.hpp"
#include "src/models/uniqueTable.hpp"
#include "src/models/qmdd.hpp"
#include "src/common/mathUtils.hpp"
#include "src/common/calculation.hpp"
#include "src/common/monitor.hpp"

using namespace std;

void execute() {
    UniqueTable& uniqueTable = UniqueTable::getInstance();
    // QMDDEdge firstEdge = mathUtils::kron(state::Ket0().getInitialEdge(), state::Ket0().getInitialEdge());
    // QuantumCircuit circuit(2, QMDDState(firstEdge));
    // // circuit.addI(0);
    // // circuit.addI(1);
    // circuit.addSWAP(1, 0);
    // queue<QMDDGate> gateQueue = circuit.getGateQueue();
    // cout << "gateQueue:" << endl;
    // printQueue(gateQueue);
    // circuit.execute();

    // cout << "finalState:" << circuit.getFinalState().getInitialEdge() << endl;


    // QMDDGate zeroGate = gate::O();

    // QMDDGate i1Gate = gate::I();
    // QMDDGate i2Gate = gate::I();
    // QMDDGate phGate = gate::Ph(0.5);
    // QMDDGate cx1Gate = gate::CX1();
    // QMDDGate cx2Gate = gate::CX2();
    // QMDDGate sGate = gate::S();
    // QMDDGate toffGate = gate::Toff();
    // QMDDGate ffredkinGate = gate::fFredkin();
    QMDDState ket0State1 = state::Ket0();
    QMDDState ket0State2 = state::Ket0();
    // QMDDState bra0State = state::Bra0();
    // cout << "zeroGate:" << zeroGate.getInitialEdge() << endl;
    // cout << "zeroGate:" << zeroGate << endl;
    // cout << "igate:" << iGate.getInitialEdge() << endl;
    // cout << "phgate:" << phGate.getInitialEdge() << endl;
    // cout << "cx1gate:" << *cx1Gate.getStartNode() << endl;
    // cout << "cx2gate:" << cx2Gate.getDepth() << endl;
    // cout << "igate:" << gate::I().getInitialEdge() << endl;
    cout << "cx2gate:" << cx1Gate.getInitialEdge() << endl;
    // QMDDGate hGate = gate::H();
    // cout << "hgate:" << hGate.getInitialEdge() << endl;
    // cout << "ket0" << ket0State.getInitialEdge().uniqueTableKey << endl;
    // cout << "bra0" << bra0State.getInitialEdge().uniqueTableKey << endl;
    QMDDState ket0State = mathUtils::kron(ket0State1.getInitialEdge(), ket0State2.getInitialEdge());
    vector<complex<double>> ket0Elements = ket0State.getInitialEdge().getAllElementsForKet();
    for (size_t i = 0; i < ket0Elements.size(); i++) {
        cout << "ket0: "<< ket0Elements[i] << endl;
    }
    // cout << "bra0:" << bra0State.getStartNode()->edges.size() << ", " << bra0State.getStartNode()->edges[0].size() << endl;
    // QMDDGate xGate = gate::X();
    // cout << "xgate:" << xGate.getInitialEdge() << endl;
    // QMDDState ket0 = state::Ket0();
    // QMDDState bra0 = state::Bra0();
    // auto rebk1 = mathUtils::kron(ket0.getInitialEdge(), bra0.getInitialEdge());
    // auto rebk2 = mathUtils::kron(ket0.getInitialEdge(), bra0.getInitialEdge());
    // auto result1 = mathUtils::mul(cx1Gate.getInitialEdge(), cx2Gate.getInitialEdge());
    // auto result2 = mathUtils::add(cx1Gate.getInitialEdge(), cx2Gate.getInitialEdge());
    // auto result3 = mathUtils::add(xGate.getInitialEdge(), hGate.getInitialEdge());
    // cout << "result1:" << result1 << endl;
    // cout << "result2:" << result2 << endl;
    // cout << "rebk1:" << rebk1 << endl;
    // cout << "rebk2:" << rebk2 << endl;
    // QMDDGate swap2 = gate::SWAP(true);
    // cout << "swap2:" << swap2.getInitialEdge() << endl;
    // QMDDGate swap = gate::SWAP();
    // cout << "swap:" << swap.getInitialEdge() << endl;
    uniqueTable.printAllEntries();


    // QMDDGate cx1 = gate::CX1();
    // QMDDGate cx2 = gate::CX2();
    // auto result2 = mathUtils::add(cx1.getInitialEdge(), cx2.getInitialEdge());
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

