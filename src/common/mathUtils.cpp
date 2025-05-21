#include "mathUtils.hpp"

QMDDEdge mathUtils::mul(const QMDDEdge& e0, const QMDDEdge& e1, int depth) {
    jniUtils& cache = jniUtils::getInstance();
    long long operationCacheKey = calculation::generateOperationCacheKey(OperationKey(e0, OperationType::MUL, e1));
    OperationResult existingAnswer = cache.jniFind(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        QMDDEdge answer = QMDDEdge(existingAnswer.first, existingAnswer.second);
        if (answer.getStartNode() != nullptr) {
            // cout << "\033[1;36mCache hit!\033[0m" << endl;
            return answer;
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

    vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size(), QMDDEdge(.0, nullptr)));
    complex<double> tmpWeight = .0;
    bool allWeightsAreZero = true;
    if (depth < CONFIG.process.parallelism){
        // cout << "\033[1;31mmulti thread mul\033[0m" << endl;
        vector<future<QMDDEdge>> futures;
        size_t futureIdx = 0;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n1->edges[i].size(); j++) {
                futures.push_back(threadPool.enqueue([&, i, j]() {
                    QMDDEdge answer = QMDDEdge(.0, nullptr);
                    for (size_t k = 0; k < n0->edges[0].size(); k++) {
                        QMDDEdge p(e0.weight * n0->edges[i][k].weight, n0->edges[i][k].uniqueTableKey);
                        QMDDEdge q(e1.weight * n1->edges[k][j].weight, n1->edges[k][j].uniqueTableKey);
                        answer = mathUtils::add(answer, mathUtils::mul(p, q, depth + 1), depth + 1);
                    }
                    return answer;
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

    // } else if (depth < CONFIG.process.parallelism + CONFIG.process.concurrency){
    //     // cout << "\033[1;34mmulti fiber mul\033[0m" << endl;

    //     vector<boost::fibers::future<QMDDEdge>> futures;

    //     for (int i = 0; i < n0->edges.size(); i++) {
    //         for (int j = 0; j < n1->edges[0].size(); j++) {
    //             futures.emplace_back(
    //                 boost::fibers::async([&, i, j]() {
    //                     QMDDEdge answer = QMDDEdge(.0, nullptr);
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

    } else{
        // cout << "sequential mul" << endl;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n1->edges[0].size(); j++){
                for (size_t k = 0; k < n0->edges[0].size(); k++) {
                    QMDDEdge p(e0.weight * n0->edges[i][k].weight, n0->edges[i][k].uniqueTableKey);
                    QMDDEdge q(e1.weight * n1->edges[k][j].weight, n1->edges[k][j].uniqueTableKey);
                    z[i][j] = mathUtils::add(z[i][j], mathUtils::mul(p, q, depth + 1), depth + 1);
                }
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
        result = QMDDEdge(.0, nullptr);
    } else {
        result = QMDDEdge(tmpWeight, make_shared<QMDDNode>(z));
    }
    cache.jniInsert(operationCacheKey, result.weight, result.uniqueTableKey);
    return result;
}

QMDDEdge mathUtils::mulForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {

    jniUtils& cache = jniUtils::getInstance();
    long long operationCacheKey = calculation::generateOperationCacheKey(
        OperationKey(e0, OperationType::MUL, e1)
    );

    auto cacheFuture = threadPool.enqueue([&cache, operationCacheKey]() -> QMDDEdge {
        OperationResult existing = cache.jniFind(operationCacheKey);
        if (existing != OperationResult{.0, 0}) {
            QMDDEdge answer{ existing.first, existing.second };
            if (answer.getStartNode() != nullptr) {
                return answer;
            }
        }
        return edgeZero;
    });

    auto computeFuture = threadPool.enqueue([=]() -> QMDDEdge {
        if (e1.isTerminal) std::swap(const_cast<QMDDEdge&>(e0), const_cast<QMDDEdge&>(e1));
        if (e0.isTerminal) {
            if (e0.weight == .0)         return e0;
            else if (e0.weight == 1.0)   return e1;
            else                         return QMDDEdge(e0.weight * e1.weight, e1.uniqueTableKey);
        }
        auto n0 = e0.getStartNode();
        auto n1 = e1.getStartNode();
        vector<vector<QMDDEdge>> z(2, std::vector<QMDDEdge>(2, edgeZero));
        complex<double> tmpWeight = .0;
        bool allZero = true;

        for (size_t n = 0; n < 2; n++) {
            QMDDEdge p(e0.weight * n0->edges[n][n].weight, n0->edges[n][n].uniqueTableKey);
            QMDDEdge q(e1.weight * n1->edges[n][n].weight, n1->edges[n][n].uniqueTableKey);
            z[n][n] = mathUtils::mulForDiagonal(p, q);
            if (z[n][n].weight != .0) {
                allZero = false;
                if (tmpWeight == .0) {
                    tmpWeight = z[n][n].weight;
                    z[n][n].weight = 1.0;
                } else {
                    z[n][n].weight /= tmpWeight;
                }
            }
        }

        QMDDEdge result;
        if (allZero) {
            result = edgeZero;
        } else {
            result = QMDDEdge(e0.weight * tmpWeight, std::make_shared<QMDDNode>(z));
        }
        return result;
    });


    while (true) {
        if (cacheFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            QMDDEdge cached = cacheFuture.get();
            if (cached != edgeZero) {
                return cached;
            }

            QMDDEdge computed = computeFuture.get();

            threadPool.enqueue([&cache, operationCacheKey, computed]() {
                cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
            });
            return computed;
        }

        if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            QMDDEdge computed = computeFuture.get();
            threadPool.enqueue([&cache, operationCacheKey, computed]() {
                cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
            });
            return computed;
        }
    }
}

QMDDEdge mathUtils::add(const QMDDEdge& e0, const QMDDEdge& e1, int depth) {
    jniUtils& cache = jniUtils::getInstance();
    long long operationCacheKey = calculation::generateOperationCacheKey(OperationKey(e0, OperationType::ADD, e1));
    OperationResult existingAnswer = cache.jniFind(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        QMDDEdge answer = QMDDEdge(existingAnswer.first, existingAnswer.second);
        if (answer.getStartNode() != nullptr) {
            // cout << "\033[1;36mCache hit!\033[0m" << endl;
            return answer;
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
            return QMDDEdge(e0.weight + e1.weight, nullptr);
        }
    }
    shared_ptr<QMDDNode> n0 = e0.getStartNode();
    shared_ptr<QMDDNode> n1 = e1.getStartNode();
    bool allWeightsAreZero = true;
    vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n0->edges[0].size()));
    complex<double> tmpWeight = .0;
    if (depth < CONFIG.process.parallelism){
        // cout << "\033[1;31mmulti thread add\033[0m" << endl;
        vector<future<QMDDEdge>> futures;
        size_t futureIdx = 0;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                futures.push_back(threadPool.enqueue([&, i, j]() {
                    QMDDEdge p(e0.weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
                    QMDDEdge q(e1.weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
                    if (q.isTerminal) {
                        std::swap(p, q);
                    }
                    if (p.isTerminal) {
                        if (p.weight == .0) {
                            return q;
                        } else if (q.isTerminal) {
                            return QMDDEdge(p.weight + q.weight, nullptr);
                        }
                    }
                    if (p.uniqueTableKey == q.uniqueTableKey) {
                        return QMDDEdge(p.weight + q.weight, p.uniqueTableKey);
                    } else {
                        return mathUtils::add(p, q, depth + 1);
                    }
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
    // } else if (depth < CONFIG.process.parallelism + CONFIG.process.concurrency){
    //     // cout << "\033[1;34mmulti fiber add\033[0m" << endl;

    //     vector<boost::fibers::future<QMDDEdge>> futures;

    //     for (int i = 0; i < n0->edges.size(); i++) {
    //         for (int j = 0; j < n0->edges[i].size(); j++) {
    //             futures.emplace_back(
    //                 boost::fibers::async([&, i, j]() {
    //                     QMDDEdge p(e0.weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
    //                     QMDDEdge q(e1.weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
    //                     if (q.isTerminal) {
    //                         std::swap(p, q);
    //                     }
    //                     if (p.isTerminal) {
    //                         if (p.weight == .0) {
    //                             return q;
    //                         } else if (q.isTerminal) {
    //                             return QMDDEdge(p.weight + q.weight, nullptr);
    //                         }
    //                     }
    //                     if (p.uniqueTableKey == q.uniqueTableKey) {
    //                         return QMDDEdge(p.weight + q.weight, p.uniqueTableKey);
    //                     }else {
    //                         return mathUtils::add(p, q, depth + 1);
    //                     }
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

    } else{
        // cout << "sequential add" << endl;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                QMDDEdge p(e0.weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
                QMDDEdge q(e1.weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
                if (q.isTerminal) {
                    std::swap(p, q);
                }
                if (p.isTerminal) {
                    if (p.weight == .0) {
                        z[i][j] = q;
                    } else if (q.isTerminal) {
                        z[i][j] = QMDDEdge(p.weight + q.weight, nullptr);
                    }
                }
                if (p.uniqueTableKey == q.uniqueTableKey) {
                    z[i][j] = QMDDEdge(p.weight + q.weight, p.uniqueTableKey);
                } else {
                    z[i][j] = mathUtils::add(p, q, depth + 1);
                }
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
        result = QMDDEdge(.0, nullptr);
    } else {
        result = QMDDEdge(tmpWeight, make_shared<QMDDNode>(z));
    }
    cache.jniInsert(operationCacheKey, result.weight, result.uniqueTableKey);
    return result;
}

QMDDEdge mathUtils::addForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {
    jniUtils& cache = jniUtils::getInstance();
    long long operationCacheKey = calculation::generateOperationCacheKey(
        OperationKey(e0, OperationType::ADD, e1)
    );

    auto cacheFuture = threadPool.enqueue([&cache, operationCacheKey]() -> QMDDEdge {
        OperationResult existing = cache.jniFind(operationCacheKey);
        if (existing != OperationResult{.0, 0}) {
            QMDDEdge answer{ existing.first, existing.second };
            if (answer.getStartNode() != nullptr) {
                return answer;
            }
        }
        return edgeZero;
    });

    auto computeFuture = threadPool.enqueue([=]() -> QMDDEdge {
        if (e1.isTerminal) std::swap(const_cast<QMDDEdge&>(e0), const_cast<QMDDEdge&>(e1));
        if (e0.isTerminal) {
            if (e0.weight == .0)         return e1;
            else if (e1.isTerminal)      return QMDDEdge(e0.weight + e1.weight, nullptr);
        }
        auto n0 = e0.getStartNode();
        auto n1 = e1.getStartNode();
        vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2, edgeZero));
        complex<double> tmpWeight = .0;
        bool allZero = true;

        for (size_t n = 0; n < 2; n++) {
            QMDDEdge p(e0.weight * n0->edges[n][n].weight, n0->edges[n][n].uniqueTableKey);
            QMDDEdge q(e1.weight * n1->edges[n][n].weight, n1->edges[n][n].uniqueTableKey);
            z[n][n] = mathUtils::addForDiagonal(n0->edges[n][n], e1);
            if (z[n][n].weight != .0) {
                allZero = false;
                if (tmpWeight == .0) {
                    tmpWeight = z[n][n].weight;
                    z[n][n].weight = 1.0;
                } else {
                    z[n][n].weight /= tmpWeight;
                }
            }
        }

        QMDDEdge result;
        if (allZero) {
            result = edgeZero;
        } else {
            result = QMDDEdge(e0.weight * tmpWeight, std::make_shared<QMDDNode>(z));
        }
        return result;
    });


    while (true) {
        if (cacheFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            QMDDEdge cached = cacheFuture.get();
            if (cached != edgeZero) {
                return cached;
            }

            QMDDEdge computed = computeFuture.get();

            threadPool.enqueue([&cache, operationCacheKey, computed]() {
                cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
            });
            return computed;
        }

        if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            QMDDEdge computed = computeFuture.get();
            threadPool.enqueue([&cache, operationCacheKey, computed]() {
                cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
            });
            return computed;
        }
    }
}

QMDDEdge mathUtils::kron(const QMDDEdge& e0, const QMDDEdge& e1, int depth) {
    jniUtils& cache = jniUtils::getInstance();
    long long operationCacheKey = calculation::generateOperationCacheKey(OperationKey(e0, OperationType::KRONECKER, e1));
    OperationResult existingAnswer = cache.jniFind(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        QMDDEdge answer = QMDDEdge(existingAnswer.first, existingAnswer.second);
        if (answer.getStartNode() != nullptr) {
            // cout << "\033[1;36mCache hit!\033[0m" << endl;
            return answer;
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
    // } else if (depth < CONFIG.process.parallelism + CONFIG.process.concurrency){
    //     // cout << "\033[1;34mmulti fiber kron\033[0m" << endl;
    //     vector<boost::fibers::future<QMDDEdge>> futures;
    //     for (int i = 0; i < n0->edges.size(); i++) {
    //         for (int j = 0; j < n0->edges[i].size(); j++) {
    //             futures.emplace_back(
    //                 boost::fibers::async([&, i, j]() {
    //                     return mathUtils::kron(n0->edges[i][j], e1, depth + 1);
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
        result = QMDDEdge(.0, nullptr);
    } else {
        result = QMDDEdge(e0.weight * tmpWeight, make_shared<QMDDNode>(z));
    }
    cache.jniInsert(operationCacheKey, result.weight, result.uniqueTableKey);
    return result;
}

QMDDEdge mathUtils::kronForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {

    jniUtils& cache = jniUtils::getInstance();
    long long operationCacheKey = calculation::generateOperationCacheKey(
        OperationKey(e0, OperationType::KRONECKER, e1)
    );

    auto cacheFuture = threadPool.enqueue([&cache, operationCacheKey]() -> QMDDEdge {
        OperationResult existing = cache.jniFind(operationCacheKey);
        if (existing != OperationResult{.0, 0}) {
            QMDDEdge answer{ existing.first, existing.second };
            if (answer.getStartNode() != nullptr) {
                return answer;
            }
        }
        return edgeZero;
    });

    auto computeFuture = threadPool.enqueue([=]() -> QMDDEdge {
        if (e0.isTerminal) {
            if (e0.weight == .0)         return e0;
            else if (e0.weight == 1.0)   return e1;
            else                         return QMDDEdge(e0.weight * e1.weight, e1.uniqueTableKey);
        }
        auto n0 = e0.getStartNode();
        auto n1 = e1.getStartNode();
        vector<vector<QMDDEdge>> z(2, std::vector<QMDDEdge>(2, edgeZero));
        complex<double> tmpWeight = .0;
        bool allZero = true;

        for (size_t n = 0; n < 2; n++) {
            z[n][n] = mathUtils::kronForDiagonal(n0->edges[n][n], e1);
            if (z[n][n].weight != .0) {
                allZero = false;
                if (tmpWeight == .0) {
                    tmpWeight = z[n][n].weight;
                    z[n][n].weight = 1.0;
                } else {
                    z[n][n].weight /= tmpWeight;
                }
            }
        }

        QMDDEdge result;
        if (allZero) {
            result = edgeZero;
        } else {
            result = QMDDEdge(e0.weight * tmpWeight, std::make_shared<QMDDNode>(z));
        }
        return result;
    });


    while (true) {
        if (cacheFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            QMDDEdge cached = cacheFuture.get();
            if (cached != edgeZero) {
                return cached;
            }

            QMDDEdge computed = computeFuture.get();

            threadPool.enqueue([&cache, operationCacheKey, computed]() {
                cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
            });
            return computed;
        }

        if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            QMDDEdge computed = computeFuture.get();
            threadPool.enqueue([&cache, operationCacheKey, computed]() {
                cache.jniInsert(operationCacheKey, computed.weight, computed.uniqueTableKey);
            });
            return computed;
        }
    }
}

QMDDEdge mathUtils::dyad(const QMDDEdge& e0, const QMDDEdge& e1) {
    shared_ptr<QMDDNode> n0 = e0.getStartNode();
    shared_ptr<QMDDNode> n1 = e1.getStartNode();
    // bool allWeightsAreZero = true;
    vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size()));
    // complex<double> tmpWeight = .0;
    for (size_t i = 0; i < n0->edges.size(); i++) {
        for (size_t j = 0; j < n1->edges[0].size(); j++) {
            z[i][j] = QMDDEdge(n0->edges[i][0].weight * n1->edges[0][j].weight, nullptr);
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

