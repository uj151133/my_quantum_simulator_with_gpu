#include <iostream>
#include <cstdlib>
#include <cstdlib>
#include <string>
#include <unistd.h>

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
    // cout << "Executing..." << endl;
    // vector<thread> threads;
    // mutex z_mutex;
    // for (size_t i = 0; i < 2; i++) {
    //     for (size_t j = 0; j < 2; j++) {
    //         threads.emplace_back([&, i, j]() {
    //             cout << "Creating thread [" << i << "][" << j << "]" << endl;
    //         });
    //     }
    // }
    // for (auto& thread : threads) {
    //     if (thread.joinable()) {
    //         thread.join();
    //         cout << "Thread joined" << endl;
    //     }
    // }
    
    // UniqueTable& uniqueTable = UniqueTable::getInstance();

    int numQubits = 9;
    // int numGates = 200;

    // randomRotate(numQubits, numGates);
    int omega = std::pow(2, numQubits) - 1;

    grover(numQubits, omega);
    // cout << mathUtils::mulParallel(gate::CX1().getInitialEdge(), gate::CX2().getInitialEdge()) << endl;

    // cout << mathUtils::kronParallel(gate::H().getInitialEdge(), gate::H().getInitialEdge()) << endl;

    // uniqueTable.printAllEntries();
}



int main() {
    // string processType = getProcessType();
    // if (processType == "sequential") {
    //     cout << "逐次処理を実行します。" << endl;
    //     sequentialProcessing();
    // } else if (processType == "multi-thread") {
    //     cout << "マルチスレッド処理を実行します。" << endl;
    //     parallelProcessing();
    // } else if (processType == "multi-fiber") {
    //     cout << "マルチファイバー処理を実行します。" << endl;
    //     fiberProcessing();
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

