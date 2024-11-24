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
// #include "src/test/Grover/grover.hpp"
#include "src/test/random/randomRotate.hpp"

using namespace std;

// 二重ループと再帰を組み合わせた再帰関数
int complexRecursive(int n, int m) {
    if (n <= 0 || m <= 0) return 0;
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            sum += i + j + complexRecursive(n - 1, m - 1);
        }
    }
    return sum;
}

// テスト用関数
void testRecursiveFunction() {
    const int outerLoop = 2;      // 外側ループ回数
    const int innerLoop = 2;      // 内側ループ回数
    const int recursionDepthN = 8; // 再帰の深さ (n方向)
    const int recursionDepthM = 8; // 再帰の深さ (m方向)

    std::vector<int> results(outerLoop * innerLoop);

    // OpenMP並列化
    // #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < outerLoop; ++i) {
        for (int j = 0; j < innerLoop; ++j) {
            int index = i * innerLoop + j;
            results[index] = complexRecursive(recursionDepthN, recursionDepthM);
        }
    }

    // 結果の出力
    for (int i = 0; i < std::min(10, static_cast<int>(results.size())); ++i) {
        std::cout << "結果[" << i << "] = " << results[i] << "\n";
    }
}

void execute() {

    // UniqueTable& uniqueTable = UniqueTable::getInstance();

    int numQubits = 13;
    int numGates = 700;

    randomRotate(numQubits, numGates);
    // int omega = std::pow(2, numQubits) - 1;

    // grover(numQubits, omega);
    // cout << mathUtils::mul(state::KetPlusY().getInitialEdge(), state::KetPlusY().getInitialEdge()) << endl;

    // cout << mathUtils::kron(gate::H().getInitialEdge(), gate::H().getInitialEdge()) << endl;

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

