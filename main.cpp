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


    // UniqueTable& table = UniqueTable::getInstance();
    // OperationCache& cache = OperationCache::getInstance();

    // CONFIG.printConfig();

    // QuantumCircuit circuit(14);

    // cout << "Initial state: " << circuit.getInitialState() << endl;


    int numQubits = 2;
    int numGates = 200;
  
    randomRotate(numQubits, numGates);
    // int omega = std::pow(2, numQubits) - 1;

    // grover(numQubits, omega);

    // table.printAllEntries();
    // cache.printAllEntries();
}



int main() {

    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif
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

