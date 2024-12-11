#include <iostream>
#include <cstdlib>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include "src/models/qmdd.hpp"
#include "src/common/constant.hpp"
#include "src/common/config.hpp"
#include "src/models/gate.hpp"
#include "src/models/state.hpp"
#include "src/models/uniqueTable.hpp"
#include "src/common/mathUtils.hpp"
#include "src/common/calculation.hpp"
#include "src/models/circuit.hpp"
#include "src/common/monitor.hpp"
// #include "src/test/hwb5tc/benchHwb5tc.hpp"
#include "src/test/Grover/grover.hpp"
#include "src/test/random/randomRotate.hpp"

using namespace std;

void task(int index) {
    // スレッドIDとループのインデックスを出力
    std::cout << "Thread ID: " << std::this_thread::get_id() 
              << ", Index: " << index << std::endl;
}


void execute() {
    QMDDState state01 = state::Ket0();
    QMDDState state02 = state::Ket0();
    QMDDState state03 = state::Ket0();
    QMDDState state04 = state::Ket0();
    QMDDState state05 = state::Ket0();
    QMDDState state06 = state::Ket0();
    QMDDState state07 = state::Ket0();
    QMDDState state08 = state::Ket0();
    QMDDState state09 = state::Ket0();
    QMDDState state10 = state::Ket0();

    cout << "Initial state 01: " << state01.getStartNode().get() << endl;
    cout << "Initial state 02: " << state02.getStartNode().get() << endl;
    cout << "Initial state 03: " << state03.getStartNode().get() << endl;
    cout << "Initial state 04: " << state04.getStartNode().get() << endl;
    cout << "Initial state 05: " << state05.getStartNode().get() << endl;
    cout << "Initial state 06: " << state06.getStartNode().get() << endl;
    cout << "Initial state 07: " << state07.getStartNode().get() << endl;
    cout << "Initial state 08: " << state08.getStartNode().get() << endl;
    cout << "Initial state 09: " << state09.getStartNode().get() << endl;
    cout << "Initial state 10: " << state10.getStartNode().get() << endl;


    // CONFIG.printConfig();
    UniqueTable& table = UniqueTable::getInstance();
    // table.printAllEntries();
    // QuantumCircuit circuit(14);

    // cout << "Initial state: " << circuit.getInitialState() << endl;



    int numQubits = 2;
    // int numGates = 200;

    // randomRotate(numQubits, numGates);
    // int omega = std::pow(2, numQubits) - 1;

    // grover(numQubits, omega);
    // cout << mathUtils::mul(gate::CX1().getInitialEdge(), gate::CX2().getInitialEdge()) << endl;

    // cout << mathUtils::kron(gate::H().getInitialEdge(), gate::H().getInitialEdge()) << endl;

    table.printAllEntries();
}



int main() {
    CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    // string processType = getProcessType();
    // if (processType == "sequential") {
    //     cout << "逐次処理を実行します。" << endl;
    //     sequentialProcessing();
    // } else if (processType == "multi-thread") {
    //     cout << "マルチスレッド処理を実行します。" << endl;
    //     parallelProcessing();
    // } else if (processType == "multi-fiber") {
    //     cout << "マルチファイバー処理を実行します。" << endl;
        // fiberProcessing();
    // } else {
    //     cerr << "不明な処理タイプ: " << processType << endl;
    // }
    // printMemoryUsage();
    // printMemoryUsageOnMac();

    // bool isGuiEnabled = isExecuteGui();

    // if (isGuiEnabled) {
    //     cout << "GUI is enabled." << endl;
    // } else {
    //     cout << "GUI is disabled." << endl;
    // }

    measureExecutionTime(execute);
    // execute();

    // printMemoryUsage();
    // printMemoryUsageOnMac();
    return 0;
}

