#include "mathUtils.hpp"

QMDDEdge mathUtils::mul(const QMDDEdge& e0, const QMDDEdge& e1, bool done) {
    OperationCacheClient& cache = OperationCacheClient::getInstance();
    int64_t operationCacheKey = calculation::generateOperationCacheKey(OperationKey(e0, OperationType::MUL, e1));
    if (auto existingEdge = cache.find(operationCacheKey)) {
        if (existingEdge->weight != .0 && existingEdge->uniqueTableKey != 0) {
            // cout << "\033[1;36mCache hit!\033[0m" << endl;
            return *existingEdge;
        }
    }

    if (e1.isTerminal) {
        std::swap(const_cast<QMDDEdge&>(e0), const_cast<QMDDEdge&>(e1));
    }
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

    vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size(), edgeZero));
    complex<double> tmpWeight = .0;
    bool allWeightsAreZero = true;

    vector<future<pair<pair<size_t, size_t>, QMDDEdge>>> futures;
    vector<pair<size_t, size_t>> parallelTasks;
    vector<pair<size_t, size_t>> sequentialTasks;

    for (size_t i = 0; i < n0->edges.size(); i++) {
        for (size_t j = 0; j < n1->edges[i].size(); j++) {
            if (done == false && min({n0->edges[i][0].depth, n0->edges[i][1].depth, n1->edges[0][j].depth, n1->edges[1][j].depth}) >= CONFIG.process.parallelism) {
                parallelTasks.push_back({i, j});
            } else {
                sequentialTasks.push_back({i, j});
            }
        }
    }

    for (const auto& [i, j] : parallelTasks) {
        futures.push_back(threadPool.enqueue([&, i, j]() -> pair<pair<size_t, size_t>, QMDDEdge> {
            QMDDEdge answer = edgeZero;
            for (size_t k = 0; k < n0->edges[0].size(); k++) {
                QMDDEdge p(e0.weight * n0->edges[i][k].weight, n0->edges[i][k].uniqueTableKey);
                QMDDEdge q(e1.weight * n1->edges[k][j].weight, n1->edges[k][j].uniqueTableKey);
                answer = mathUtils::add(answer, mathUtils::mul(p, q, true), true);
            }
            return {{i, j}, answer};
        }));
    }

    // 4. 逐次処理（直接z配列に格納）
    for (const auto& [i, j] : sequentialTasks) {
        QMDDEdge answer = edgeZero;
        for (size_t k = 0; k < n0->edges[0].size(); k++) {
            QMDDEdge p(e0.weight * n0->edges[i][k].weight, n0->edges[i][k].uniqueTableKey);
            QMDDEdge q(e1.weight * n1->edges[k][j].weight, n1->edges[k][j].uniqueTableKey);
            answer = mathUtils::add(answer, mathUtils::mul(p, q, true), true);
        }
        z[i][j] = answer;
    }

    // 5. 並列処理の結果を回収
    for (auto& future : futures) {
        const auto& [indices, result] = future.get();
        const auto& [i, j] = indices;
        z[i][j] = result;
    }

    // 6. 全て揃った後で重み処理
    for (size_t i = 0; i < z.size(); i++) {
        for (size_t j = 0; j < z[i].size(); j++) {
            if (z[i][j].weight != .0) {
                allWeightsAreZero = false;
                if (tmpWeight == .0) {
                    tmpWeight = z[i][j].weight;
                    z[i][j].weight = 1.0;
                } else {
                    z[i][j].weight /= tmpWeight;
                }
            }
        }
    }


    // if (depth < CONFIG.process.parallelism){
        // cout << "\033[1;31mmulti thread mul\033[0m" << endl;
        // vector<future<QMDDEdge>> futures;
        // size_t futureIdx = 0;
        // for (size_t i = 0; i < n0->edges.size(); i++) {
        //     for (size_t j = 0; j < n1->edges[i].size(); j++) {
        //         futures.push_back(threadPool.enqueue([&, i, j]() {
        //             QMDDEdge answer = QMDDEdge(.0, 0);
        //             for (size_t k = 0; k < n0->edges[0].size(); k++) {
        //                 QMDDEdge p(e0.weight * n0->edges[i][k].weight, n0->edges[i][k].uniqueTableKey);
        //                 QMDDEdge q(e1.weight * n1->edges[k][j].weight, n1->edges[k][j].uniqueTableKey);
        //                 answer = mathUtils::add(answer, mathUtils::mul(p, q, depth + 1), depth + 1);
        //             }
        //             return answer;
        //         }));
        //     }
        // }

        // futureIdx = 0;
        // for (size_t i = 0; i < z.size(); i++) {
        //     for (size_t j = 0; j < z[i].size(); j++) {
        //         z[i][j] = futures[futureIdx++].get();
        //         if (z[i][j].weight != .0) {
        //             allWeightsAreZero = false;
        //             if (tmpWeight == .0) {
        //                 tmpWeight = z[i][j].weight;
        //                 z[i][j].weight = 1.0;
        //             } else {
        //                 z[i][j].weight /= tmpWeight;
        //             }
        //         }
        //     }
        // }

    // } else if (depth < CONFIG.process.parallelism + CONFIG.process.concurrency){
    //     // cout << "\033[1;34mmulti fiber mul\033[0m" << endl;

    //     vector<boost::fibers::future<QMDDEdge>> futures;

    //     for (int i = 0; i < n0->edges.size(); i++) {
    //         for (int j = 0; j < n1->edges[0].size(); j++) {
    //             futures.emplace_back(
    //                 boost::fibers::async([&, i, j]() {
    //                     QMDDEdge answer = QMDDEdge(.0, 0);
    //                     for (size_t k = 0; k < n0->edges[0].size(); k++) {
    //                         QMDDEdge p(e0.weight * n0->edges[i][k].weight, n0->edges[i][k].uniqueTableKey);
    //                         QMDDEdge q(e1.weight * n1->edges[k][j].weight, n1->edges[k][j].uniqueTableKey);
    //                         answer = mathUtils::add(answer, mathUtils::mul(p, q, depth + 1), depth + 1);
    //                     }
    //                     return answer;
    //             }));
    //         }
    //     }

    //     size_t futureIdx = 0;
    //     for (size_t i = 0; i < z.size(); i++) {
    //         for (size_t j = 0; j < z[i].size(); j++) {
    //             z[i][j] = futures[futureIdx++].get();
    //             if (z[i][j].weight != .0) {
    //                 allWeightsAreZero = false;
    //                 if (tmpWeight == .0) {
    //                     tmpWeight = z[i][j].weight;
    //                     z[i][j].weight = 1.0;
    //                 }else if (tmpWeight != .0) {
    //                     z[i][j].weight /= tmpWeight;
    //                 } else {
    //                     cout << "⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️" << endl;
    //                 }
    //             }
    //         }
    //     }

    // } else{
    //     // cout << "sequential mul" << endl;
    //     for (size_t i = 0; i < n0->edges.size(); i++) {
    //         for (size_t j = 0; j < n1->edges[0].size(); j++){
    //             for (size_t k = 0; k < n0->edges[0].size(); k++) {
    //                 QMDDEdge p(e0.weight * n0->edges[i][k].weight, n0->edges[i][k].uniqueTableKey);
    //                 QMDDEdge q(e1.weight * n1->edges[k][j].weight, n1->edges[k][j].uniqueTableKey);
    //                 z[i][j] = mathUtils::add(z[i][j], mathUtils::mul(p, q, depth + 1), depth + 1);
    //             }
    //             if (z[i][j].weight != .0) {
    //                 allWeightsAreZero = false;
    //                 if (tmpWeight == .0) {
    //                     tmpWeight = z[i][j].weight;
    //                     z[i][j].weight = 1.0;
    //                 }else if (tmpWeight != .0) {
    //                     z[i][j].weight /= tmpWeight;
    //                 } else {
    //                     cout << "⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️" << endl;
    //                 }
    //             }
    //         }
    //     }
    // }
    QMDDEdge result;
    if (allWeightsAreZero) {
        result = edgeZero;
    } else {
        result = QMDDEdge(tmpWeight, make_shared<QMDDNode>(z));
    }
    cache.insert(operationCacheKey, result);
    return result;
}

// QMDDEdge mathUtils::mulForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {

//     jniUtils& cache = jniUtils::getInstance();
//     int64_t operationCacheKey = calculation::generateOperationCacheKey(
//         OperationKey(e0, OperationType::MUL, e1)
//     );

//     auto cacheFuture = threadPool.enqueue([&cache, operationCacheKey]() -> QMDDEdge {
//         OperationResult existing = cache.jniFind(operationCacheKey);
//         if (existing != OperationResult{.0, 0}) {
//             QMDDEdge answer{ existing.first, existing.second };
//             if (answer.uniqueTableKey!= 0) {
//                 return answer;
//             }
//         }
//         return edgeZero;
//     });

//     auto computeFuture = threadPool.enqueue([=]() -> QMDDEdge {
//         if (e1.isTerminal) std::swap(const_cast<QMDDEdge&>(e0), const_cast<QMDDEdge&>(e1));
//         if (e0.isTerminal) {
//             if (e0.weight == .0)         return e0;
//             else if (e0.weight == 1.0)   return e1;
//             else                         return QMDDEdge(e0.weight * e1.weight, e1.uniqueTableKey);
//         }
//         auto n0 = e0.getStartNode();
//         auto n1 = e1.getStartNode();
//         vector<vector<QMDDEdge>> z(2, std::vector<QMDDEdge>(2, edgeZero));
//         complex<double> tmpWeight = .0;
//         bool allZero = true;

//         for (size_t n = 0; n < 2; n++) {
//             QMDDEdge p(e0.weight * n0->edges[n][n].weight, n0->edges[n][n].uniqueTableKey);
//             QMDDEdge q(e1.weight * n1->edges[n][n].weight, n1->edges[n][n].uniqueTableKey);
//             z[n][n] = mathUtils::mulForDiagonal(p, q);
//             if (z[n][n].weight != .0) {
//                 allZero = false;
//                 if (tmpWeight == .0) {
//                     tmpWeight = z[n][n].weight;
//                     z[n][n].weight = 1.0;
//                 } else {
//                     z[n][n].weight /= tmpWeight;
//                 }
//             }
//         }

//         QMDDEdge result;
//         if (allZero) {
//             result = edgeZero;
//         } else {
//             result = QMDDEdge(e0.weight * tmpWeight, std::make_shared<QMDDNode>(z));
//         }
//         return result;
//     });


//     while (true) {
//         if (cacheFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge cached = cacheFuture.get();
//             if (cached != edgeZero) {
//                 return cached;
//             }

//             QMDDEdge computed = computeFuture.get();

//             threadPool.enqueue([&cache, operationCacheKey, computed]() {
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
//             });
//             return computed;
//         }

//         if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge computed = computeFuture.get();
//             threadPool.enqueue([&cache, operationCacheKey, computed]() {
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
//             });
//             return computed;
//         }
//     }
// }

QMDDEdge mathUtils::add(const QMDDEdge& e0, const QMDDEdge& e1, bool done) {
    OperationCacheClient& cache = OperationCacheClient::getInstance();
    int64_t operationCacheKey = calculation::generateOperationCacheKey(OperationKey(e0, OperationType::ADD, e1));
    if (auto existingEdge = cache.find(operationCacheKey)) {
        if (existingEdge->weight != .0 && existingEdge->uniqueTableKey != 0) {
            // cout << "\033[1;36mCache hit!\033[0m" << endl;
            return *existingEdge;
        }
    }
    // cout << "\033[1;35mCache miss!\033[0m" << endl;

    if (e1.isTerminal) {
        std::swap(const_cast<QMDDEdge&>(e0), const_cast<QMDDEdge&>(e1));
    }
    if (e0.isTerminal) {
        if (e0.weight == .0) {
            return e1;
        } else if (e1.isTerminal) {
            return QMDDEdge(e0.weight + e1.weight, 0);
        }
    }
    if (e0.uniqueTableKey == e1.uniqueTableKey) {
        return QMDDEdge(e0.weight + e1.weight, e0.uniqueTableKey);
    }
    shared_ptr<QMDDNode> n0 = e0.getStartNode();
    shared_ptr<QMDDNode> n1 = e1.getStartNode();
    bool allWeightsAreZero = true;
    vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n0->edges[0].size(), edgeZero));
    complex<double> tmpWeight = .0;

    vector<future<pair<pair<size_t, size_t>, QMDDEdge>>> futures;
    vector<pair<size_t, size_t>> parallelTasks;
    vector<pair<size_t, size_t>> sequentialTasks;

    // 2. タスク分類
    for (size_t i = 0; i < n0->edges.size(); i++) {
        for (size_t j = 0; j < n0->edges[i].size(); j++) {
            if (done == false && min(n0->edges[i][j].depth, n1->edges[i][j].depth) >= CONFIG.process.parallelism) {
                parallelTasks.push_back({i, j});
            } else {
                sequentialTasks.push_back({i, j});
            }
        }
    }

    for (const auto& [i, j] : parallelTasks) {
        futures.push_back(threadPool.enqueue([&, i, j]() -> pair<pair<size_t, size_t>, QMDDEdge> {
            QMDDEdge p(e0.weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
            QMDDEdge q(e1.weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
            QMDDEdge result = mathUtils::add(p, q, true);
            return {{i, j}, result};
        }));
    }

    // 4. 逐次処理（直接z配列に格納）
    for (const auto& [i, j] : sequentialTasks) {
        QMDDEdge p(e0.weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
        QMDDEdge q(e1.weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
        z[i][j] = mathUtils::add(p, q, true);
    }

    // 5. 並列処理の結果を回収
    for (auto& future : futures) {
        const auto& [indices, result] = future.get();
        const auto& [i, j] = indices;
        z[i][j] = result;
    }

    // 6. 全て揃った後で重み処理
    for (size_t i = 0; i < z.size(); i++) {
        for (size_t j = 0; j < z[i].size(); j++) {
            if (z[i][j].weight != .0) {
                allWeightsAreZero = false;
                if (tmpWeight == .0) {
                    tmpWeight = z[i][j].weight;
                    z[i][j].weight = 1.0;
                } else {
                    z[i][j].weight /= tmpWeight;
                }
            }
        }
    }

    // if (depth < CONFIG.process.parallelism){
    //     // cout << "\033[1;31mmulti thread add\033[0m" << endl;
    //     vector<future<QMDDEdge>> futures;
    //     size_t futureIdx = 0;
    //     for (size_t i = 0; i < n0->edges.size(); i++) {
    //         for (size_t j = 0; j < n0->edges[i].size(); j++) {
    //             futures.push_back(threadPool.enqueue([&, i, j]() {
    //                 QMDDEdge p(e0.weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
    //                 QMDDEdge q(e1.weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
    //                 return mathUtils::add(p, q, depth + 1);
    //             }));
    //         }
    //     }

    //     futureIdx = 0;
    //     for (size_t i = 0; i < z.size(); i++) {
    //         for (size_t j = 0; j < z[i].size(); j++) {
    //             z[i][j] = futures[futureIdx++].get();
    //             if (z[i][j].weight != .0) {
    //                 allWeightsAreZero = false;
    //                 if (tmpWeight == .0) {
    //                     tmpWeight = z[i][j].weight;
    //                     z[i][j].weight = 1.0;
    //                 } else {
    //                     z[i][j].weight /= tmpWeight;
    //                 }
    //             }
    //         }
    //     }
    // } else if (depth < CONFIG.process.parallelism + CONFIG.process.concurrency){
    //     // cout << "\033[1;34mmulti fiber add\033[0m" << endl;

    //     vector<boost::fibers::future<QMDDEdge>> futures;

    //     for (int i = 0; i < n0->edges.size(); i++) {
    //         for (int j = 0; j < n0->edges[i].size(); j++) {
    //             futures.emplace_back(
    //                 boost::fibers::async([&, i, j]() {
    //                     QMDDEdge p(e0.weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
    //                     QMDDEdge q(e1.weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
    //                     return mathUtils::add(p, q, depth + 1);
    //             }));
    //         }
    //     }

    //     size_t futureIdx = 0;
    //     for (size_t i = 0; i < z.size(); i++) {
    //         for (size_t j = 0; j < z[i].size(); j++) {
    //             z[i][j] = futures[futureIdx++].get();
    //             if (z[i][j].weight != .0) {
    //                 allWeightsAreZero = false;
    //                 if (tmpWeight == .0) {
    //                     tmpWeight = z[i][j].weight;
    //                     z[i][j].weight = 1.0;
    //                 }else if (tmpWeight != .0) {
    //                     z[i][j].weight /= tmpWeight;
    //                 } else {
    //                     cout << "⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️" << endl;
    //                 }
    //             }
    //         }
    //     }

    // } else{
    //     // cout << "sequential add" << endl;
    //     for (size_t i = 0; i < n0->edges.size(); i++) {
    //         for (size_t j = 0; j < n0->edges[i].size(); j++) {
    //             QMDDEdge p(e0.weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
    //             QMDDEdge q(e1.weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
    //             z[i][j] = mathUtils::add(p, q, depth + 1);
    //             if (z[i][j].weight != .0) {
    //                 allWeightsAreZero = false;
    //                 if (tmpWeight == .0) {
    //                     tmpWeight = z[i][j].weight;
    //                     z[i][j].weight = 1.0;
    //                 }else if (tmpWeight != .0) {
    //                     z[i][j].weight /= tmpWeight;
    //                 } else {
    //                     cout << "⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️" << endl;
    //                 }
    //             }
    //         }
    //     }
    // }
    QMDDEdge result;
    if (allWeightsAreZero) {
        result = edgeZero;
    } else {
        result = QMDDEdge(tmpWeight, make_shared<QMDDNode>(z));
    }
    cache.insert(operationCacheKey, result);
    return result;
}

// QMDDEdge mathUtils::addForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {
//     jniUtils& cache = jniUtils::getInstance();
//     int64_t operationCacheKey = calculation::generateOperationCacheKey(
//         OperationKey(e0, OperationType::ADD, e1)
//     );

//     auto cacheFuture = threadPool.enqueue([&cache, operationCacheKey]() -> QMDDEdge {
//         OperationResult existing = cache.jniFind(operationCacheKey);
//         if (existing != OperationResult{.0, 0}) {
//             QMDDEdge answer{ existing.first, existing.second };
//             if (answer.uniqueTableKey != 0) {
//                 return answer;
//             }
//         }
//         return edgeZero;
//     });

//     auto computeFuture = threadPool.enqueue([=]() -> QMDDEdge {
//         if (e1.isTerminal) std::swap(const_cast<QMDDEdge&>(e0), const_cast<QMDDEdge&>(e1));
//         if (e0.isTerminal) {
//             if (e0.weight == .0)         return e1;
//             else if (e1.isTerminal)      return QMDDEdge(e0.weight + e1.weight, 0);
//         }
//         auto n0 = e0.getStartNode();
//         auto n1 = e1.getStartNode();
//         vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2, edgeZero));
//         complex<double> tmpWeight = .0;
//         bool allZero = true;

//         for (size_t n = 0; n < 2; n++) {
//             QMDDEdge p(e0.weight * n0->edges[n][n].weight, n0->edges[n][n].uniqueTableKey);
//             QMDDEdge q(e1.weight * n1->edges[n][n].weight, n1->edges[n][n].uniqueTableKey);
//             z[n][n] = mathUtils::addForDiagonal(n0->edges[n][n], e1);
//             if (z[n][n].weight != .0) {
//                 allZero = false;
//                 if (tmpWeight == .0) {
//                     tmpWeight = z[n][n].weight;
//                     z[n][n].weight = 1.0;
//                 } else {
//                     z[n][n].weight /= tmpWeight;
//                 }
//             }
//         }

//         QMDDEdge result;
//         if (allZero) {
//             result = edgeZero;
//         } else {
//             result = QMDDEdge(e0.weight * tmpWeight, std::make_shared<QMDDNode>(z));
//         }
//         return result;
//     });


//     while (true) {
//         if (cacheFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge cached = cacheFuture.get();
//             if (cached != edgeZero) {
//                 return cached;
//             }

//             QMDDEdge computed = computeFuture.get();

//             threadPool.enqueue([&cache, operationCacheKey, computed]() {
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
//             });
//             return computed;
//         }

//         if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge computed = computeFuture.get();
//             threadPool.enqueue([&cache, operationCacheKey, computed]() {
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
//             });
//             return computed;
//         }
//     }
// }

QMDDEdge mathUtils::kron(const QMDDEdge& e0, const QMDDEdge& e1, int depth) {
    OperationCacheClient cache = OperationCacheClient::getInstance();
    int64_t operationCacheKey = calculation::generateOperationCacheKey(OperationKey(e0, OperationType::KRONECKER, e1));
    if (auto existingEdge = cache.find(operationCacheKey)) {
        if (existingEdge->weight != .0 && existingEdge->uniqueTableKey != 0) {
            // cout << "\033[1;36mCache hit!\033[0m" << endl;
            return *existingEdge;
        }
    }
    // cout << "\033[1;35mCache miss!\033[0m" << endl;

    if (e0.isTerminal) {
        if (e0.weight == .0) {
            return e0;
        }else if (e0.weight == 1.0) {
            return e1;
        } else {
            return QMDDEdge(e0.weight * e1.weight, e1.uniqueTableKey);
        }
    }
    shared_ptr<QMDDNode> n0 = e0.getStartNode();
    shared_ptr<QMDDNode> n1 = e1.getStartNode();
    vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size()));
    complex<double> tmpWeight = .0;
    bool allWeightsAreZero = true;
    if (depth < CONFIG.process.parallelism){
        vector<future<QMDDEdge>> futures;
        size_t futureIdx = 0;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                futures.push_back(threadPool.enqueue([&, i, j]() {
                    return mathUtils::kron(n0->edges[i][j], e1, depth + 1);
                }));
            }
        }

        futureIdx = 0;
        for (size_t i = 0; i < z.size(); i++) {
            for (size_t j = 0; j < z[i].size(); j++) {
                z[i][j] = futures[futureIdx++].get();
                if (z[i][j].weight != .0) {
                    allWeightsAreZero = false;
                    if (tmpWeight == .0) {
                        tmpWeight = z[i][j].weight;
                        z[i][j].weight = 1.0;
                    } else {
                        z[i][j].weight /= tmpWeight;
                    }
                }
            }
        }
    } else if (depth < CONFIG.process.parallelism + CONFIG.process.concurrency){
        // cout << "\033[1;34mmulti fiber kron\033[0m" << endl;
        vector<boost::fibers::future<QMDDEdge>> futures;
        for (int i = 0; i < n0->edges.size(); i++) {
            for (int j = 0; j < n0->edges[i].size(); j++) {
                futures.emplace_back(
                    boost::fibers::async([&, i, j]() {
                        return mathUtils::kron(n0->edges[i][j], e1, depth + 1);
                }));
            }
        }

        size_t futureIdx = 0;
        for (size_t i = 0; i < z.size(); i++) {
            for (size_t j = 0; j < z[i].size(); j++) {
                z[i][j] = futures[futureIdx++].get();
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

    } else{
        // cout << "sequential kron" << endl;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                z[i][j] = mathUtils::kron(n0->edges[i][j], e1, depth + 1);

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
    }
    QMDDEdge result;
    if (allWeightsAreZero) {
        result = edgeZero;
    } else {
        result = QMDDEdge(e0.weight * tmpWeight, make_shared<QMDDNode>(z));
    }
    cache.insert(operationCacheKey, result);
    return result;
}

QMDDEdge mathUtils::kron(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCacheClient& cache = OperationCacheClient::getInstance();
    int64_t operationCacheKey = calculation::generateOperationCacheKey(OperationKey(e0, OperationType::KRONECKER, e1));
    if (auto existingEdge = cache.find(operationCacheKey)) {
        if (existingEdge->weight != .0 && existingEdge->uniqueTableKey != 0) {
            // cout << "\033[1;36mCache hit!\033[0m" << endl;
            return *existingEdge;
        }
    }
    // cout << "\033[1;35mCache miss!\033[0m" << endl;

    if (e0.isTerminal) {
        if (e0.weight == .0) {
            return e0;
        }else if (e0.weight == 1.0) {
            return e1;
        } else {
            return QMDDEdge(e0.weight * e1.weight, e1.uniqueTableKey);
        }
    }
    shared_ptr<QMDDNode> n0 = e0.getStartNode();
    shared_ptr<QMDDNode> n1 = e1.getStartNode();
    vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size()));
    for (size_t i = 0; i < n0->edges.size(); i++) {
        for (size_t j = 0; j < n0->edges[i].size(); j++) {
            z[i][j] = QMDDEdge(n0->edges[i][j].weight, e1.uniqueTableKey);
        }
    }

    QMDDEdge result = QMDDEdge(e0.weight * e1.weight, make_shared<QMDDNode>(z));
    cache.insert(operationCacheKey, result);
    return result;
}


// QMDDEdge mathUtils::kronForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {

//     jniUtils& cache = jniUtils::getInstance();
//     int64_t operationCacheKey = calculation::generateOperationCacheKey(
//         OperationKey(e0, OperationType::KRONECKER, e1)
//     );

//     auto cacheFuture = threadPool.enqueue([&cache, operationCacheKey]() -> QMDDEdge {
//         OperationResult existing = cache.jniFind(operationCacheKey);
//         if (existing != OperationResult{.0, 0}) {
//             QMDDEdge answer{ existing.first, existing.second };
//             if (answer.uniqueTableKey != 0) {
//                 return answer;
//             }
//         }
//         return edgeZero;
//     });

//     auto computeFuture = threadPool.enqueue([=]() -> QMDDEdge {
//         if (e0.isTerminal) {
//             if (e0.weight == .0)         return e0;
//             else if (e0.weight == 1.0)   return e1;
//             else                         return QMDDEdge(e0.weight * e1.weight, e1.uniqueTableKey);
//         }
//         auto n0 = e0.getStartNode();
//         auto n1 = e1.getStartNode();
//         vector<vector<QMDDEdge>> z(2, std::vector<QMDDEdge>(2, edgeZero));
//         complex<double> tmpWeight = .0;
//         bool allZero = true;

//         for (size_t n = 0; n < 2; n++) {
//             z[n][n] = mathUtils::kronForDiagonal(n0->edges[n][n], e1);
//             if (z[n][n].weight != .0) {
//                 allZero = false;
//                 if (tmpWeight == .0) {
//                     tmpWeight = z[n][n].weight;
//                     z[n][n].weight = 1.0;
//                 } else {
//                     z[n][n].weight /= tmpWeight;
//                 }
//             }
//         }

//         QMDDEdge result;
//         if (allZero) {
//             result = edgeZero;
//         } else {
//             result = QMDDEdge(e0.weight * tmpWeight, std::make_shared<QMDDNode>(z));
//         }
//         return result;
//     });


//     while (true) {
//         if (cacheFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge cached = cacheFuture.get();
//             if (cached != edgeZero) {
//                 return cached;
//             }

//             QMDDEdge computed = computeFuture.get();

//             // threadPool.enqueue([&cache, operationCacheKey, computed]() {
//             //     cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
//             // });
//             return computed;
//         }

//         if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge computed = computeFuture.get();
//             // threadPool.enqueue([&cache, operationCacheKey, computed]() {
//             //     cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
//             // });
//             return computed;
//         }
//     }
// }

QMDDEdge mathUtils::dyad(const QMDDEdge& e0, const QMDDEdge& e1) {
    shared_ptr<QMDDNode> n0 = e0.getStartNode();
    shared_ptr<QMDDNode> n1 = e1.getStartNode();
    vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size()));
    for (size_t i = 0; i < n0->edges.size(); i++) {
        for (size_t j = 0; j < n1->edges[0].size(); j++) {
            z[i][j] = QMDDEdge(n0->edges[i][0].weight * n1->edges[0][j].weight, 0);
        }
    }
    QMDDEdge result;
    result = QMDDEdge(1.0, make_shared<QMDDNode>(z));
    return result;
}


double mathUtils::csc(const double theta) {
    double sin_theta = sin(theta);
    if (sin_theta == .0) throw overflow_error("csc(θ) is undefined (sin(θ) = 0)");
    return 1.0 / sin_theta;
}

complex<double> mathUtils::csc(const complex<double> theta) {
    complex<double> sin_theta = sin(theta);
    if (sin_theta == .0) throw overflow_error("csc(θ) is undefined (sin(θ) = 0)");
    return 1.0 / sin_theta;
}

double mathUtils::sec(const double theta) {
    double cos_theta = cos(theta);
    if (cos_theta == .0) throw overflow_error("sec(θ) is undefined (cos(θ) = 0)");
    return 1.0 / cos_theta;
}

complex<double> mathUtils::sec(const complex<double> theta) {
    complex<double> cos_theta = cos(theta);
    if (cos_theta == .0) throw overflow_error("sec(θ) is undefined (cos(θ) = 0)");
    return 1.0 / cos_theta;
}

double mathUtils::cot(const double theta) {
    double tan_theta = tan(theta);
    if (tan_theta == .0) throw overflow_error("cot(θ) is undefined (tan(θ) = 0)");
    return 1.0 / tan_theta;
}

complex<double> mathUtils::cot(const complex<double> theta) {
    complex<double> tan_theta = tan(theta);
    if (tan_theta == .0) throw overflow_error("cot(θ) is undefined (tan(θ) = 0)");
    return 1.0 / tan_theta;
}

// double mathUtils::sumOfSquares(const vector<complex<double>>& vec) {
//     return accumulate(vec.begin(), vec.end(), 0.0, [](double sum, const complex<double>& val) {
//         return sum + std::pow(abs(val), 2);
//     });
// }

double mathUtils::sumOfSquares(const vector<complex<double>>& vec) {
    return accumulate(vec.begin(), vec.end(), 0.0, [](double sum, const complex<double>& val) {
        return sum + std::pow(abs(val), 2);
    });
}

vector<int> mathUtils::createRange(int start, int end) {
    int min = std::min(start, end);
    int max = std::max(start, end);
    vector<int> range;

    for (int i = min; i <= max; ++i) {
        range.push_back(i);
    }
    return range;
}
