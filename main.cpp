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

// 二重ループと再帰を組み合わせた再帰関数
QMDDEdge complexRecursive(int n, int m, QMDDEdge e0, QMDDEdge e1) {
    OperationCache& cache = OperationCache::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::TEST, e1));
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    QMDDEdge Zero(.0, nullptr);
    if (n <= 0 || m <= 0 ) return Zero;
    if (e1.isTerminal) swap(e0, e1);
    if (e0.isTerminal) {
            if (e0.weight == .0) {
                return e0;
            } else if (e0.weight == 1.0){
                return e1;
            } else {
                return QMDDEdge(e0.weight * e1.weight, e1.uniqueTableKey);
            }
        }

    shared_ptr<QMDDNode> n0 = e0.getStartNode();
    shared_ptr<QMDDNode> n1 = e1.getStartNode();

    vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size(), Zero));
    complex<double> tmpWeight = .0;
    bool allWeightsAreZero = true;
    for (int i = 0; i < n0->edges.size(); ++i) {
        for (int j = 0; j < n0->edges[0].size(); ++j) {
            z[i][j] = complexRecursive(n - 1, m - 1, e0, e1);

            if (z[i][j].weight != .0) {
                allWeightsAreZero = false;
                if (tmpWeight == .0) {
                    tmpWeight = z[i][j].weight;
                    z[i][j].weight = 1.0;
                }else if (tmpWeight != .0) {
                    z[i][j].weight /= tmpWeight;
                } else {
                    cout << "⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️" << endl;
                }
            }
        }
    }
    QMDDEdge result;
    if (allWeightsAreZero) {
        result = QMDDEdge(.0, nullptr);
    } else {
        result = QMDDEdge(tmpWeight, make_shared<QMDDNode>(z));
    }
    cache.insert(operationCacheKey, make_pair(tmpWeight, result.uniqueTableKey));
    return result;
}

// テスト用関数
void testRecursiveFunction() {
    const int recursionDepthN = 6; // 再帰の深さ (n方向)
    const int recursionDepthM = 6; // 再帰の深さ (m方向)

    QMDDEdge edge = mathUtils::kron(gate::Toff().getInitialEdge(), gate::Toff().getInitialEdge());
    // QMDDEdge edge = gate::CX1().getInitialEdge();
    shared_ptr<QMDDNode> node = edge.getStartNode();
    int outerLoop = node->edges.size();
    int innerLoop = node->edges[0].size();

    std::vector<QMDDEdge> results(outerLoop * innerLoop);
    vector<vector<QMDDEdge>> z(outerLoop, vector<QMDDEdge>(innerLoop));

    // OpenMP並列化
    // #pragma omp parallel for schedule(dynamic) default(shared)
    for (int i = 0; i < outerLoop; ++i) {
        for (int j = 0; j < innerLoop; ++j) {
            QMDDEdge p(edge.weight * node->edges[i][j].weight, node->edges[i][j].uniqueTableKey);
            QMDDEdge q(edge.weight * node->edges[j][i].weight, node->edges[j][i].uniqueTableKey);
            z[i][j] = complexRecursive(recursionDepthN, recursionDepthM, p, q);
        }
    }
}

void execute() {
    for (int i = 0; i < 10; i++){
        testRecursiveFunction();
    }
    
    // UniqueTable& uniqueTable = UniqueTable::getInstance();

    int numQubits = 11;
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

