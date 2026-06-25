#include "mathUtils.hpp"

QMDDEdge mathUtils::mul(const QMDDEdge& e0, const QMDDEdge& e1, bool onFiber, int depth) {
    // cout << "\033[1;34mEntering mul: depth=" << depth << " e0.weight=" << e0.weight << " e0.sonKind_=" << static_cast<int>(e0.sonKind_) << " e0.key_=" << e0.key_ << " e1.weight=" << e1.weight << " e1.sonKind_=" << static_cast<int>(e1.sonKind_) << " e1.key_=" << e1.key_ << "\033[0m" << endl;
    QMDDEdge a = e0;
    QMDDEdge b = e1;
    if (b.isTerminal) std::swap(a, b);
    if (a.sonKind_ == SonKind::Terminal) {
        if (a.magnitude == -numeric_limits<double>::infinity()) {
            return a;
        } else if (a.magnitude == 0.0 && a.angle == 0.0) {
            return b;
        } else {
            return QMDDEdge(
                {a.magnitude + b.magnitude, a.angle + b.angle},
                b.key_,
                b.son_
            );
        }
    }
    // cout << "\033[1;34mEntering mul: depth=" << depth << " a.weight=" << a.weight << " b.weight=" << b.weight << "\033[0m" << endl;
    if (depth >= PARAMETER.parallelism.GPU || a.sonKind_ == SonKind::SVLeaf || b.sonKind_ == SonKind::SVLeaf) {
        // cout << "\033[1;34mEntering mul: depth=" << depth << " a.weight=" << a.weight << " a.sonKind_=" << static_cast<int>(a.sonKind_) << " a.key_=" << a.key_ << " b.weight=" << b.weight << " b.sonKind_=" << static_cast<int>(b.sonKind_) << " b.key_=" << b.key_ << "\033[0m" << endl;
        const int goalDepth = max(a.depth, b.depth);
        
        if (a.depth < goalDepth) a = mathUtils::beAuthentic(a, goalDepth);
        if (b.depth < goalDepth) b = mathUtils::beAuthentic(b, goalDepth);
        
        GPUInput A = farewell(a);
        GPUInput B = farewell(b);

        void* outRe = nullptr;
        void* outIm = nullptr;

        int64_t outId = 0;
        gpu_precision outCoef[2] = {.0f, .0f};

        runMulAny2Wrapper(A, B, &outRe, &outIm, &outId, outCoef);

        if (outCoef[0] == .0f && outCoef[1] == .0f) {
            releaseGpuBuffer(outRe);
            releaseGpuBuffer(outIm);
            return edgeZero;
        }
        
        return QMDDEdge(
            toLogPolar(complex<double>(outCoef[0], outCoef[1])),
            outId,
            make_shared<SVLeaf>(A.dim, outRe, outIm)
        );
    }

    bool concurrency = depth < PARAMETER.parallelism.fiber;

    int64_t operationCacheKey = calculation::generateOperationCacheKey(OperationKey(a.key_, b.key_));
    OperationCacheClient& cache = OperationCacheClient::getInstance();
    if (PARAMETER.cache.alive) {
        if (auto existingEdge = cache.find(operationCacheKey, onFiber)) {
            if (existingEdge->magnitude != -numeric_limits<double>::infinity() && existingEdge->key_ != 0) {
                return QMDDEdge(
                    {
                        existingEdge->magnitude + a.magnitude + b.magnitude,
                        existingEdge->angle + a.angle + b.angle
                    },
                    existingEdge->getSon()
                );
            }
        }
    }

    // cout << "\033[1;35mCache miss!\033[0m" << endl;

    shared_ptr<QMDDNode> n0 = get<shared_ptr<QMDDNode>>(a.getSon());
    shared_ptr<QMDDNode> n1 = get<shared_ptr<QMDDNode>>(b.getSon());

    vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2, edgeZero));
    

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
                            QMDDEdge p({n0->edges[i][k].magnitude, n0->edges[i][k].angle}, n0->edges[i][k].getSon());
                            QMDDEdge q({n1->edges[k][j].magnitude, n1->edges[k][j].angle}, n1->edges[k][j].getSon());
                            answer = mathUtils::add(answer, mathUtils::mul(p, q, true, depth + 1), depth + 1);
                        }
                        return {{i, j}, answer};
                    })
                );
            } else {
                QMDDEdge answer = edgeZero;
                for (size_t k = 0; k < 2; k++) {
                    QMDDEdge p({n0->edges[i][k].magnitude, n0->edges[i][k].angle}, n0->edges[i][k].getSon());
                    QMDDEdge q({n1->edges[k][j].magnitude, n1->edges[k][j].angle}, n1->edges[k][j].getSon());
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

    // for (size_t i = 0; i < 2; i++) {
    //     for (size_t j = 0; j < 2; j++) {
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
    bool allWeightsAreZero = true;
    pair<double, double> tmpWeight = normalize(z, allWeightsAreZero);

    if (PARAMETER.cache.alive) {
        cache.insert(operationCacheKey, QMDDEdge(tmpWeight, make_shared<QMDDNode>(z)), onFiber);
    }

    QMDDEdge result = allWeightsAreZero ? edgeZero : QMDDEdge(
        {a.magnitude + b.magnitude + tmpWeight.first, a.angle + b.angle + tmpWeight.second},
        make_shared<QMDDNode>(z)
    );
    return result;
}


// QMDDEdge mathUtils::mulForDiagonal(const QMDDEdge& a, const QMDDEdge& b) {

//     jniUtils& cache = jniUtils::getInstance();
//     int64_t operationCacheKey = calculation::generateOperationCacheKey(
//         OperationKey(a, OperationType::MUL, b)
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
//         if (b.isTerminal) std::swap(const_cast<QMDDEdge&>(a), const_cast<QMDDEdge&>(b));
//         if (a.isTerminal) {
//             if (a.weight == .0)         return a;
//             else if (a.weight == 1.0)   return b;
//             else                         return QMDDEdge(a.weight * b.weight, b.key_);
//         }
//         auto n0 = a.getSon();
//         auto n1 = b.getSon();
//         vector<vector<QMDDEdge>> z(2, std::vector<QMDDEdge>(2, edgeZero));
//         complex<double> tmpWeight = .0;
//         bool allZero = true;

//         for (size_t n = 0; n < 2; n++) {
//             QMDDEdge p(a.weight * n0->edges[n][n].weight, n0->edges[n][n].key_);
//             QMDDEdge q(b.weight * n1->edges[n][n].weight, n1->edges[n][n].key_);
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
//             result = QMDDEdge(a.weight * tmpWeight, make_shared<QMDDNode>(z));
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
    QMDDEdge a = e0;
    QMDDEdge b = e1;
    if (b.isTerminal) std::swap(a, b);
    if (a.isTerminal) {
        if (a.magnitude == -numeric_limits<double>::infinity()) {
            return b;
        } else if (b.isTerminal) {
            return QMDDEdge(toLogPolar(toComplex(a.magnitude, a.angle) + toComplex(b.magnitude, b.angle)));
        }
    }
    if (a.key_ == b.key_) {
        return QMDDEdge(
            toLogPolar(toComplex(a.magnitude, a.angle) + toComplex(b.magnitude, b.angle)),
            a.key_,
            a.son_
        );
    }
    if (depth >= PARAMETER.parallelism.GPU || a.sonKind_ == SonKind::SVLeaf || b.sonKind_ == SonKind::SVLeaf) {

        const int goalDepth = max(a.depth, b.depth);
        
        if (a.depth < goalDepth) a = mathUtils::beAuthentic(a, goalDepth);
        if (b.depth < goalDepth) b = mathUtils::beAuthentic(b, goalDepth);
    
        GPUInput A = farewell(a);
        GPUInput B = farewell(b);

        void* outRe = nullptr;
        void* outIm = nullptr;

        int64_t outId = 0;
        gpu_precision outCoef[2] = {.0f, .0f};

        runAddAny2Wrapper(A, B, &outRe, &outIm, &outId, outCoef);

        if (outCoef[0] == .0f && outCoef[1] == .0f) {
            releaseGpuBuffer(outRe);
            releaseGpuBuffer(outIm);
            return edgeZero;
        }
        
        return QMDDEdge(
            toLogPolar(complex<double>(outCoef[0], outCoef[1])),
            outId,
            make_shared<SVLeaf>(A.dim, outRe, outIm)
        );
    }

    bool concurrency = depth < PARAMETER.parallelism.fiber;

    shared_ptr<QMDDNode> n0 = get<shared_ptr<QMDDNode>>(a.getSon());
    shared_ptr<QMDDNode> n1 = get<shared_ptr<QMDDNode>>(b.getSon());
    vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2, edgeZero));


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
                        QMDDEdge p(
                            {a.magnitude + n0->edges[i][j].magnitude, a.angle + n0->edges[i][j].angle},
                            n0->edges[i][j].getSon()
                        );
                        QMDDEdge q(
                            {b.magnitude + n1->edges[i][j].magnitude, b.angle + n1->edges[i][j].angle},
                            n1->edges[i][j].getSon()
                        );
                        QMDDEdge r = mathUtils::add(p, q, depth + 1);
                        return {{i, j}, r};
                    })
                );
            } else {
                QMDDEdge p(
                    {a.magnitude + n0->edges[i][j].magnitude, a.angle + n0->edges[i][j].angle},
                    n0->edges[i][j].getSon()
                );
                QMDDEdge q(
                    {b.magnitude + n1->edges[i][j].magnitude, b.angle + n1->edges[i][j].angle},
                    n1->edges[i][j].getSon()
                );
                z[i][j] = mathUtils::add(p, q, depth + 1);
            }
        }
    }

    for (auto& ff : fiberFutures) {
        const auto& [indices, result] = ff.get();
        const auto& [i, j] = indices;
        z[i][j] = result;
    }

    // for (size_t i = 0; i < 2; i++) {
    //     for (size_t j = 0; j < 2; j++) {
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
    bool allWeightsAreZero = true;

    pair<double, double> tmpWeight = normalize(z, allWeightsAreZero);

    QMDDEdge result = allWeightsAreZero ? edgeZero : QMDDEdge(tmpWeight, make_shared<QMDDNode>(z));
    return result;
}

// QMDDEdge mathUtils::addForDiagonal(const QMDDEdge& a, const QMDDEdge& b) {
//     jniUtils& cache = jniUtils::getInstance();
//     int64_t operationCacheKey = calculation::generateOperationCacheKey(
//         OperationKey(a, OperationType::ADD, b)
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
//         if (b.isTerminal) std::swap(const_cast<QMDDEdge&>(a), const_cast<QMDDEdge&>(b));
//         if (a.isTerminal) {
//             if (a.weight == .0)         return b;
//             else if (b.isTerminal)      return QMDDEdge(a.weight + b.weight, 0);
//         }
//         auto n0 = a.getSOn();
//         auto n1 = b.getSon();
//         vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2, edgeZero));
//         complex<double> tmpWeight = .0;
//         bool allZero = true;

//         for (size_t n = 0; n < 2; n++) {
//             QMDDEdge p(a.weight * n0->edges[n][n].weight, n0->edges[n][n].key_);
//             QMDDEdge q(b.weight * n1->edges[n][n].weight, n1->edges[n][n].key_);
//             z[n][n] = mathUtils::addForDiagonal(n0->edges[n][n], b);
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
//             result = QMDDEdge(a.weight * tmpWeight, make_shared<QMDDNode>(z));
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
        if (e0.magnitude == -numeric_limits<double>::infinity()) {
            return e0;
        } else if (e0.magnitude == 0.0 && e0.angle == 0.0) {
            return e1;
        } else {
            return QMDDEdge(
                {e0.magnitude + e1.magnitude, e0.angle + e1.angle},
                e1.key_,
                e1.son_
            );
        }
    }

    if (depth >= PARAMETER.parallelism.GPU || e0.sonKind_ == SonKind::SVLeaf) {
        GPUInput A = farewell(e0);
        GPUInput B = farewell(e1);

        void* outRe = nullptr;
        void* outIm = nullptr;
        int64_t outId = 0;
        gpu_precision outCoef[2] = {.0f, .0f};

        runKronAny2Wrapper(A, B, &outRe, &outIm, &outId, outCoef);

        if (outCoef[0] == .0f && outCoef[1] == .0f) {
            releaseGpuBuffer(outRe);
            releaseGpuBuffer(outIm);
            return edgeZero;
        }
        
        return QMDDEdge(
            toLogPolar(complex<double>(outCoef[0], outCoef[1])),
            outId,
            make_shared<SVLeaf>(A.dim * B.dim, outRe, outIm)
        );
    }
    shared_ptr<QMDDNode> n0 = get<shared_ptr<QMDDNode>>(e0.getSon());

    vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            z[i][j] = mathUtils::kron(n0->edges[i][j], e1, depth + 1);

            // if (z[i][j].weight != .0) {
            //     allWeightsAreZero = false;
            //     if (tmpWeight == .0) {
            //         tmpWeight = z[i][j].weight;
            //         z[i][j].weight = 1.0;
            //     }else if (tmpWeight != .0) {
            //         z[i][j].weight /= tmpWeight;
            //     } else {
            //         cout << "⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️" << endl;
            //     }
            // }
        }
    }

    bool allWeightsAreZero = true;
    
    pair<double, double> tmpWeight = normalize(z, allWeightsAreZero);
    QMDDEdge result = allWeightsAreZero ? edgeZero : QMDDEdge(
        {e0.magnitude + tmpWeight.first, e0.angle + tmpWeight.second},
        make_shared<QMDDNode>(z)
    );
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
        return QMDDEdge({e0.magnitude + e1.magnitude, e0.angle + e1.angle});
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
    result = QMDDEdge(toLogPolar(complex<double>(1.0, 0.0)), make_shared<QMDDNode>(z));
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

double mathUtils::logSumExp(double logA, double logB) {
    if (!std::isfinite(logA) || logA == -numeric_limits<double>::infinity()) return logB;
    if (!std::isfinite(logB) || logB == -numeric_limits<double>::infinity()) return logA;
    if (logA > logB) return logA + std::log1p(std::exp(logB - logA));
    return logB + std::log1p(std::exp(logA - logB));
}

double mathUtils::logWeightNormSq(double scaledNormSq, double logScale) {
    if (scaledNormSq <= 0.0) {
        return -numeric_limits<double>::infinity();
    }
    if (logScale == 0.0) {
        return std::log(scaledNormSq);
    }
    return 2.0 * logScale + std::log(scaledNormSq);
}

double mathUtils::probabilityFromLogWeights(double logP0, double logP1) {
    const double logSum = logSumExp(logP0, logP1);
    if (!std::isfinite(logSum) || logSum == -numeric_limits<double>::infinity()) {
        return 0.0;
    }
    return std::exp(logP0 - logSum);
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

pair<double, double> mathUtils::toLogPolar(const complex<double>& w) {
    if (isZERO(w)) {
        return {-numeric_limits<double>::infinity(), numeric_limits<double>::quiet_NaN()};
    }
    double angle = arg(w);
    if (!isnan(angle)) {
        angle = remainder(angle, 2.0 * M_PI);
    }
    return {log(abs(w)), angle};
}

complex<double> mathUtils::toComplex(double magnitude, double angle) {
    if (magnitude == -numeric_limits<double>::infinity()) {
        return {0.0, 0.0};
    }
    return polar(exp(magnitude), angle);
}

pair<double, double> mathUtils::normalize(vector<vector<QMDDEdge>>& e, bool& allWeightsAreZero) {
    allWeightsAreZero = true;
    pair<double, double> tmpWeight = {
        -numeric_limits<double>::infinity(),
        numeric_limits<double>::quiet_NaN()
    };

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (e[i][j].magnitude == -numeric_limits<double>::infinity()) {
                continue;
            }

            allWeightsAreZero = false;

            if (tmpWeight.first == -numeric_limits<double>::infinity()) {
                tmpWeight = {e[i][j].magnitude, e[i][j].angle};
                e[i][j].magnitude = 0.0;
                e[i][j].angle = 0.0;
            } else {
                e[i][j].magnitude -= tmpWeight.first;
                e[i][j].angle = remainder(e[i][j].angle - tmpWeight.second, 2.0 * M_PI);
            }
        }
    }

    return tmpWeight;
}


QMDDEdge mathUtils::beAuthentic(const QMDDEdge& e, int end) {
    QMDDEdge out = e;
    while (out.depth < end) {
        out = mathUtils::kron(out, identityEdge, PARAMETER.parallelism.GPU);
    }
    return out;
}