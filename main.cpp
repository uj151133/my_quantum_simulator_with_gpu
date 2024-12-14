#include <iostream>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <getopt.h>


#include "src/common/config.hpp"
#include "src/models/qmdd.hpp"
#include "src/common/constant.hpp"
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


void execute() {
    // CONFIG.printConfig();
    UniqueTable& table = UniqueTable::getInstance();
    // table.printAllEntries();
    // QuantumCircuit circuit(14);

    // cout << "Initial state: " << circuit.getInitialState() << endl;

    // QuantumCircuit circuit(20);
    // circuit.addRx(9, 2.61854);
    // cout << "finish adding rx gate" << endl;

    // QMDDEdge edge = QMDDEdge(1.0, 13421976852400610425);
    // QMDDGate gate1 = mathUtils::kron(gate::I().getInitialEdge(), gate::I().getInitialEdge());
    // cout << "gate1: " << gate1.getInitialEdge().uniqueTableKey << endl;
    // cout << "gate1: " << gate1.getStartNode() << endl;
    // QMDDEdge edge = QMDDEdge(1.0, shared_ptr<QMDDNode>(gate1.getStartNode()));
    // cout << "edge: " << edge << endl;
    // circuit.addRz(6, 1.49602);

    int numQubits = 2;
    int numGates = 200;

    randomRotate(numQubits, numGates);
    // int omega = std::pow(2, numQubits) - 1;

    // grover(numQubits, omega);
    // cout << mathUtils::mul(gate::CX1().getInitialEdge(), gate::CX2().getInitialEdge()) << endl;

    // cout << mathUtils::kron(gate::H().getInitialEdge(), gate::H().getInitialEdge()) << endl;

    // table.printAllEntries();
}



int main() {

    cout << "entry count: " << ENTRY_COUNT << endl;

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

