#include "mathUtils.hpp"

QMDDEdge mathUtils::mul(const QMDDEdge& e0, const QMDDEdge& e1, bool onFiber, int depth) {
    // cout << "\033[1;34mEntering mul: depth=" << depth << " e0.weight=" << e0.weight << " e0.sonKind_=" << static_cast<int>(e0.sonKind_) << " e0.key_=" << e0.key_ << " e1.weight=" << e1.weight << " e1.sonKind_=" << static_cast<int>(e1.sonKind_) << " e1.key_=" << e1.key_ << "\033[0m" << endl;
    if (e1.isTerminal) {
        std::swap(const_cast<QMDDEdge&>(e0), const_cast<QMDDEdge&>(e1));
    }
    if (e0.sonKind_ == SonKind::Terminal) {
        if (e0.weight == .0) {
            return e0;
        } else if (e0.weight == 1.0){
            return e1;
        } else {
            return QMDDEdge(e0.weight * e1.weight, e1.key_, e1.son_);
        }
    }
    // cout << "\033[1;34mEntering mul: depth=" << depth << " e0.weight=" << e0.weight << " e1.weight=" << e1.weight << "\033[0m" << endl;
    if (depth >= PARAMETER.parallelism.GPU || e0.sonKind_ == SonKind::SVLeaf || e1.sonKind_ == SonKind::SVLeaf) {
        // cout << "\033[1;34mEntering mul: depth=" << depth << " e0.weight=" << e0.weight << " e0.sonKind_=" << static_cast<int>(e0.sonKind_) << " e0.key_=" << e0.key_ << " e1.weight=" << e1.weight << " e1.sonKind_=" << static_cast<int>(e1.sonKind_) << " e1.key_=" << e1.key_ << "\033[0m" << endl;
        GPUInput A = farewell(e0);
        GPUInput B = farewell(e1);

        void* outRe = nullptr;
        void* outIm = nullptr;

        int64_t outId = 0;
        gpu_precision outCoef[2] = {.0f, .0f};

        runMulAny2Wrapper(A, B,
                        &outRe, &outIm,
                        &outId, outCoef);
        // std::cerr << "GPU mul result: outId=" << outId << " outCoef=(" << outCoef[0] << "," << outCoef[1] << ")\n";
        return QMDDEdge(complex<double>(outCoef[0], outCoef[1]), outId, make_shared<SVLeaf>(A.dim, outRe, outIm));
    }

    bool concurrency = depth < PARAMETER.parallelism.fiber;

    int64_t operationCacheKey = calculation::generateOperationCacheKey(OperationKey(e0.key_, e1.key_));
    OperationCacheClient& cache = OperationCacheClient::getInstance();
    if (PARAMETER.cache.alive) {
        if (auto existingEdge = cache.find(operationCacheKey, onFiber)) {
            if (existingEdge->weight != .0 && existingEdge->key_ != 0) {
                QMDDEdge result = QMDDEdge(existingEdge->weight * e0.weight * e1.weight, existingEdge->getSon());
                return result;
            }
        }
    }

    // cout << "\033[1;35mCache miss!\033[0m" << endl;

    shared_ptr<QMDDNode> n0 = get<shared_ptr<QMDDNode>>(e0.getSon());
    shared_ptr<QMDDNode> n1 = get<shared_ptr<QMDDNode>>(e1.getSon());

    vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2, edgeZero));
    complex<double> tmpWeight = .0;
    bool allWeightsAreZero = true;

    vector<boost::fibers::future<pair<pair<size_t, size_t>, QMDDEdge>>> fiberFutures;

    // cout << "Preparing tasks for multiplication: " << n0->edges.size() << " x " << n1->edges[0].size() << endl;

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            array<int, 4> depths{
                n0->edges[i][0].depth,
                n0->edges[i][1].depth,
                n1->edges[0][j].depth,
                n1->edges[1][j].depth
            };
            double calculatedDepth = mathUtils::median(depths);
            if (concurrency) {
                fiberFutures.emplace_back(
                    boost::fibers::async([&, i, j]() -> pair<pair<size_t, size_t>, QMDDEdge> {
                        QMDDEdge answer = edgeZero;
                        for (size_t k = 0; k < 2; k++) {
                            QMDDEdge p(n0->edges[i][k].weight, n0->edges[i][k].getSon());
                            QMDDEdge q(n1->edges[k][j].weight, n1->edges[k][j].getSon());
                            answer = mathUtils::add(answer, mathUtils::mul(p, q, true, depth + 1), depth + 1);
                        }
                        return {{i, j}, answer};
                    })
                );
            } else {
                QMDDEdge answer = edgeZero;
                for (size_t k = 0; k < 2; k++) {
                    QMDDEdge p(n0->edges[i][k].weight, n0->edges[i][k].getSon());
                    QMDDEdge q(n1->edges[k][j].weight, n1->edges[k][j].getSon());
                    answer = mathUtils::add(answer, mathUtils::mul(p, q, onFiber, depth + 1), depth + 1);
                }
                z[i][j] = answer;
            }
        }
    }



    for (auto& ff : fiberFutures) {
        const auto& [indices, result] = ff.get();
        const auto& [i, j] = indices;
        z[i][j] = result;
    }

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
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

    if (PARAMETER.cache.alive) {
        cache.insert(operationCacheKey, QMDDEdge(tmpWeight, make_shared<QMDDNode>(z)), onFiber);
    }
    QMDDEdge result = allWeightsAreZero ? edgeZero : QMDDEdge(e0.weight * e1.weight * tmpWeight, make_shared<QMDDNode>(z));
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
//             if (answer.key_!= 0) {
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
//             else                         return QMDDEdge(e0.weight * e1.weight, e1.key_);
//         }
//         auto n0 = e0.getSon();
//         auto n1 = e1.getSon();
//         vector<vector<QMDDEdge>> z(2, std::vector<QMDDEdge>(2, edgeZero));
//         complex<double> tmpWeight = .0;
//         bool allZero = true;

//         for (size_t n = 0; n < 2; n++) {
//             QMDDEdge p(e0.weight * n0->edges[n][n].weight, n0->edges[n][n].key_);
//             QMDDEdge q(e1.weight * n1->edges[n][n].weight, n1->edges[n][n].key_);
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
//             result = QMDDEdge(e0.weight * tmpWeight, make_shared<QMDDNode>(z));
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
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             });
//             return computed;
//         }

//         if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge computed = computeFuture.get();
//             threadPool.enqueue([&cache, operationCacheKey, computed]() {
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             });
//             return computed;
//         }
//     }
// }

QMDDEdge mathUtils::add(const QMDDEdge& e0, const QMDDEdge& e1, int depth) {
    if (e1.isTerminal) {
        std::swap(const_cast<QMDDEdge&>(e0), const_cast<QMDDEdge&>(e1));
    }
    if (e0.isTerminal) {
        if (e0.weight == .0) {
            return e1;
        } else if (e1.isTerminal) {
            return QMDDEdge(e0.weight + e1.weight);
        }
    }
    if (e0.key_ == e1.key_) {
        return QMDDEdge(e0.weight + e1.weight, e0.key_, e0.son_);
    }
    if (depth >= PARAMETER.parallelism.GPU || e0.sonKind_ == SonKind::SVLeaf || e1.sonKind_ == SonKind::SVLeaf) {
        GPUInput A = farewell(e0);
        GPUInput B = farewell(e1);

        void* outRe = nullptr;
        void* outIm = nullptr;

        int64_t outId = 0;
        gpu_precision outCoef[2] = {.0f, .0f};

        runAddAny2Wrapper(A, B,
                        &outRe, &outIm,
                        &outId, outCoef);
        std::cerr << "GPU add result: outId=" << outId << " outCoef=(" << outCoef[0] << "," << outCoef[1] << ")\n";
        return QMDDEdge(complex<double>(outCoef[0], outCoef[1]), outId, make_shared<SVLeaf>(A.dim, outRe, outIm));
    }

    bool concurrency = depth < PARAMETER.parallelism.fiber;

    shared_ptr<QMDDNode> n0 = get<shared_ptr<QMDDNode>>(e0.getSon());
    shared_ptr<QMDDNode> n1 = get<shared_ptr<QMDDNode>>(e1.getSon());
    bool allWeightsAreZero = true;
    vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2, edgeZero));
    complex<double> tmpWeight = .0;

    vector<boost::fibers::future<pair<pair<size_t, size_t>, QMDDEdge>>> fiberFutures;

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            array<int, 2> depths{
                n0->edges[i][j].depth,
                n1->edges[i][j].depth
            };
            if (concurrency) {
                fiberFutures.emplace_back(
                    boost::fibers::async([&, i, j]() -> pair<pair<size_t, size_t>, QMDDEdge> {
                        QMDDEdge p(e0.weight * n0->edges[i][j].weight, n0->edges[i][j].getSon());
                        QMDDEdge q(e1.weight * n1->edges[i][j].weight, n1->edges[i][j].getSon());
                        QMDDEdge r = mathUtils::add(p, q, depth + 1);
                        return {{i, j}, r};
                    })
                );
            } else {
                QMDDEdge p(e0.weight * n0->edges[i][j].weight, n0->edges[i][j].getSon());
                QMDDEdge q(e1.weight * n1->edges[i][j].weight, n1->edges[i][j].getSon());
                z[i][j] = mathUtils::add(p, q, depth + 1);
            }
        }
    }

    for (auto& ff : fiberFutures) {
        const auto& [indices, result] = ff.get();
        const auto& [i, j] = indices;
        z[i][j] = result;
    }

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
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

    QMDDEdge result = allWeightsAreZero ? edgeZero : QMDDEdge(tmpWeight, make_shared<QMDDNode>(z));
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
//             if (answer.key_ != 0) {
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
//         auto n0 = e0.getSOn();
//         auto n1 = e1.getSon();
//         vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2, edgeZero));
//         complex<double> tmpWeight = .0;
//         bool allZero = true;

//         for (size_t n = 0; n < 2; n++) {
//             QMDDEdge p(e0.weight * n0->edges[n][n].weight, n0->edges[n][n].key_);
//             QMDDEdge q(e1.weight * n1->edges[n][n].weight, n1->edges[n][n].key_);
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
//             result = QMDDEdge(e0.weight * tmpWeight, make_shared<QMDDNode>(z));
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
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             });
//             return computed;
//         }

//         if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge computed = computeFuture.get();
//             threadPool.enqueue([&cache, operationCacheKey, computed]() {
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             });
//             return computed;
//         }
//     }
// }

QMDDEdge mathUtils::kron(const QMDDEdge& e0, const QMDDEdge& e1, int depth) {
    if (e0.isTerminal) {
        if (e0.weight == .0) {
            return e0;
        }else if (e0.weight == 1.0) {
            return e1;
        } else {
            return QMDDEdge(e0.weight * e1.weight, e1.key_, e1.son_);
        }
    }

    if (depth >= PARAMETER.parallelism.GPU || e0.sonKind_ == SonKind::SVLeaf) {
        GPUInput A = farewell(e0);
        GPUInput B = farewell(e1);

        void* outRe = nullptr;
        void* outIm = nullptr;
        int64_t outId = 0;
        gpu_precision outCoef[2] = {.0f, .0f};

        runKronAny2Wrapper(A, B,
                        &outRe, &outIm,
                        &outId, outCoef);
        // std::cerr << "GPU kron result: outId=" << outId << " outCoef=(" << outCoef[0] << "," << outCoef[1] << ")\n";
        return QMDDEdge(complex<double>(outCoef[0], outCoef[1]), outId, make_shared<SVLeaf>(A.dim * B.dim, outRe, outIm));
    }
    shared_ptr<QMDDNode> n0 = get<shared_ptr<QMDDNode>>(e0.getSon());

    if (!n0) { std::cerr << "kron: n0 is null (key=" << e0.key_ << ", weight= " << e0.weight << ")\n"; }

    vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2));
    complex<double> tmpWeight = .0;
    bool allWeightsAreZero = true;

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
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

    QMDDEdge result = allWeightsAreZero ? edgeZero : QMDDEdge(e0.weight * tmpWeight, make_shared<QMDDNode>(z));
    return result;
}

// QMDDEdge mathUtils::kron(const QMDDEdge& e0, const QMDDEdge& e1) {
//     if (e0.isTerminal) {
//         if (e0.weight == .0) {
//             return e0;
//         }else if (e0.weight == 1.0) {
//             return e1;
//         } else {
//             return QMDDEdge(e0.weight * e1.weight, e1.key_);
//         }
//     }
//     shared_ptr<QMDDNode> n0 = e0.getSon();
//     shared_ptr<QMDDNode> n1 = e1.getSon();
//     vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size()));
//     for (size_t i = 0; i < n0->edges.size(); i++) {
//         for (size_t j = 0; j < n0->edges[i].size(); j++) {
//             z[i][j] = QMDDEdge(n0->edges[i][j].weight, e1.key_);
//         }
//     }

//     QMDDEdge result = QMDDEdge(e0.weight * e1.weight, make_shared<QMDDNode>(z));
//     return result;
// }


// QMDDEdge mathUtils::kronForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {

//     jniUtils& cache = jniUtils::getInstance();
//     int64_t operationCacheKey = calculation::generateOperationCacheKey(
//         OperationKey(e0, OperationType::KRONECKER, e1)
//     );

//     auto cacheFuture = threadPool.enqueue([&cache, operationCacheKey]() -> QMDDEdge {
//         OperationResult existing = cache.jniFind(operationCacheKey);
//         if (existing != OperationResult{.0, 0}) {
//             QMDDEdge answer{ existing.first, existing.second };
//             if (answer.key_ != 0) {
//                 return answer;
//             }
//         }
//         return edgeZero;
//     });

//     auto computeFuture = threadPool.enqueue([=]() -> QMDDEdge {
//         if (e0.isTerminal) {
//             if (e0.weight == .0)         return e0;
//             else if (e0.weight == 1.0)   return e1;
//             else                         return QMDDEdge(e0.weight * e1.weight, e1.key_);
//         }
//         auto n0 = e0.getSon();
//         auto n1 = e1.getSon();
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
//             result = QMDDEdge(e0.weight * tmpWeight, make_shared<QMDDNode>(z));
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
//             //     cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             // });
//             return computed;
//         }

//         if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge computed = computeFuture.get();
//             // threadPool.enqueue([&cache, operationCacheKey, computed]() {
//             //     cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             // });
//             return computed;
//         }
//     }
// }

QMDDEdge mathUtils::dyad(const QMDDEdge& e0, const QMDDEdge& e1) {
    if (e0.isTerminal || e1.isTerminal) {
        return QMDDEdge(e0.weight * e1.weight);
    }
    shared_ptr<QMDDNode> n0 = get<shared_ptr<QMDDNode>>(e0.getSon());
    shared_ptr<QMDDNode> n1 = get<shared_ptr<QMDDNode>>(e1.getSon());
    vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2));
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            z[i][j] = mathUtils::dyad(n0->edges[i][0], n1->edges[0][j]);
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

int mathUtils::findCoprimeBelow(int N) {
    static thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<int> dis(2, N - 1);

    while (true) {
        int x = dis(gen);
        if (gcd(x, N) == 1) {
            return x;
        }
    }
}

bool mathUtils::isMultiplePI(double theta, double eps) {
    double quotient = theta / M_PI;
    double roundedQuotient = round(quotient);
    return abs(quotient - roundedQuotient) < eps;
}

bool mathUtils::isZERO(const complex<double>& z) {
    return z.real() == .0 && z.imag() == .0;
}