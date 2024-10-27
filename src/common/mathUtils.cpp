#include "mathUtils.hpp"

QMDDEdge mathUtils::mul(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    UniqueTable& table = UniqueTable::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::MUL, e1));
    cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        cout << "\033[1;35mCache miss!\033[0m" << endl;
        shared_ptr<QMDDNode> n0 = table.find(e0.uniqueTableKey);
        shared_ptr<QMDDNode> n1 = table.find(e1.uniqueTableKey);

        QMDDEdge* e0Copy = const_cast<QMDDEdge*>(&e0);
        QMDDEdge* e1Copy = const_cast<QMDDEdge*>(&e1);
        if (e1Copy->isTerminal) {
            QMDDEdge* tmpEdge = e0Copy;
            e0Copy = e1Copy;
            e1Copy = tmpEdge;
        }
        if (e0Copy->isTerminal) {
            if (e0Copy->weight == .0) {
                return *e0Copy;
            } else if (e0Copy->weight == 1.0){
                return * e1Copy;
            } else {
                return QMDDEdge(e0Copy->weight * e1Copy->weight, n1);
            }
        }

        vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size(), QMDDEdge(.0, nullptr)));
        complex<double> tmpWeight = .0;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n1->edges[0].size(); j++){
                for (size_t k = 0; k < n0->edges[0].size(); k++) {
                    QMDDEdge p(e0Copy->weight * n0->edges[i][k].weight, table.find(n0->edges[i][k].uniqueTableKey));
                    QMDDEdge q(e1Copy->weight * n1->edges[k][j].weight, table.find(n1->edges[k][j].uniqueTableKey));
                    z[i][j] = add(z[i][j], mul(p, q));
                }
                if (z[i][j].weight != .0 && tmpWeight == .0) {
                    tmpWeight = z[i][j].weight;
                    z[i][j].weight = 1.0;
                }else if (z[i][j].weight != .0 && tmpWeight != .0) {
                    z[i][j].weight /= tmpWeight;
                } else {
                    if (z[i][j].weight != .0) {
                        cout << "⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️" << endl;
                    }
                }

            }
        }
        auto newNode = make_shared<QMDDNode>(z);
        cache.insert(operationCacheKey, make_pair(tmpWeight, calculation::generateUniqueTableKey(*newNode)));
        return QMDDEdge(tmpWeight, newNode);
    }

}

QMDDEdge mathUtils::add(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    UniqueTable& table = UniqueTable::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::ADD, e1));
    cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        cout << "\033[1;35mCache miss!\033[0m" << endl;
        shared_ptr<QMDDNode> n0 = table.find(e0.uniqueTableKey);
        shared_ptr<QMDDNode> n1 = table.find(e1.uniqueTableKey);
        QMDDEdge* e0Copy = const_cast<QMDDEdge*>(&e0);
        QMDDEdge* e1Copy = const_cast<QMDDEdge*>(&e1);
        if (e1Copy->isTerminal) {
            QMDDEdge* tmpEdge = e0Copy;
            e0Copy = e1Copy;
            e1Copy = tmpEdge;
        }
        if (e0Copy->isTerminal) {
            if (e0Copy->weight == .0) {
                return *e1Copy;
            } else if (e1Copy->isTerminal) {
                return QMDDEdge(e0Copy->weight + e1Copy->weight, nullptr);
            }
        }

        vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n0->edges[0].size()));
        complex<double> tmpWeight = .0;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                QMDDEdge p(e0Copy->weight * n0->edges[i][j].weight, table.find(n0->edges[i][j].uniqueTableKey));
                QMDDEdge q(e1Copy->weight * n1->edges[i][j].weight, table.find(n1->edges[i][j].uniqueTableKey));
                z[i][j] = add(p, q);

                if (z[i][j].weight != .0 && tmpWeight == .0) {
                    tmpWeight = z[i][j].weight;
                    z[i][j].weight = 1.0;
                }else if (z[i][j].weight != .0 && tmpWeight != .0) {
                    z[i][j].weight /= tmpWeight;
                } else {
                    if (z[i][j].weight != .0) {
                        cout << "⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️" << endl;
                    }
                }

            }
        }
        auto newNode = make_shared<QMDDNode>(z);
        cache.insert(operationCacheKey, make_pair(tmpWeight, calculation::generateUniqueTableKey(*newNode)));
        return QMDDEdge(tmpWeight, newNode);
    }
}

QMDDEdge mathUtils::kron(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    UniqueTable& table = UniqueTable::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::KRONECKER, e1));
    cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        cout << "\033[1;35mCache miss!\033[0m" << endl;
        shared_ptr<QMDDNode> n0 = table.find(e0.uniqueTableKey);
        shared_ptr<QMDDNode> n1 = table.find(e1.uniqueTableKey);
        QMDDEdge* e0Copy = const_cast<QMDDEdge*>(&e0);
        QMDDEdge* e1Copy = const_cast<QMDDEdge*>(&e1);
        if (e0Copy->isTerminal) {
            if (e0Copy->weight == .0) {
                return *e0Copy;
            }else if (e0Copy->weight == 1.0) {
                return *e1Copy;
            } else {
                return QMDDEdge(e0Copy->weight * e1Copy->weight, n1);
            }
        }
        vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n0->edges[0].size()));
        complex<double> tmpWeight = .0;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                z[i][j] = kron(n0->edges[i][j], e1);

                if (z[i][j].weight != .0 && tmpWeight == .0) {
                    tmpWeight = z[i][j].weight;
                    z[i][j].weight = 1.0;
                }else if (z[i][j].weight != .0 && tmpWeight != .0) {
                    z[i][j].weight /= tmpWeight;
                } else {
                    if (z[i][j].weight != .0) {
                        cout << "⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️" << endl;
                    }
                }

            }
        }
        auto newNode = make_shared<QMDDNode>(z);
        cache.insert(operationCacheKey, make_pair(tmpWeight, calculation::generateUniqueTableKey(*newNode)));
        return QMDDEdge(tmpWeight, newNode);
    }
}

complex<double> mathUtils::csc(const complex<double> theta) {
    complex<double> sin_theta = sin(theta);
    if (sin_theta == .0) throw overflow_error("csc(θ) is undefined (sin(θ) = 0)");
    return 1.0 / sin_theta;
}

complex<double> mathUtils::sec(const complex<double> theta) {
    complex<double> cos_theta = cos(theta);
    if (cos_theta == .0) throw overflow_error("sec(θ) is undefined (cos(θ) = 0)");
    return 1.0 / cos_theta;
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
#if defined(__x86_64__) || defined(_M_X64)
    __m256d sum = _mm256_setzero_pd();
    
    // SIMDで処理できる部分（2つの複素数ごと = 4つのdouble）
    size_t i = 0;
    for (; i + 1 < vec.size(); i += 2) {
        __m256d v = _mm256_loadu_pd(reinterpret_cast<const double*>(&vec[i]));
        __m256d squared = _mm256_mul_pd(v, v);  // 要素ごとの平方
        sum = _mm256_add_pd(sum, squared);      // 累積加算
    }

    // SIMDレジスタから部分和を取り出す
    double result[4];
    _mm256_storeu_pd(result, sum);

    // スカラー演算で残りの1つの複素数を処理（奇数サイズの場合のみ）
    double scalarSum = result[0] + result[1] + result[2] + result[3];
    if (i < vec.size()) {
        scalarSum += pow(abs(vec[i]), 2);  // 最後の要素の絶対値の二乗を加算
    }

    return scalarSum;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    float64x2_t sum = vdupq_n_f64(0.0);  // 64ビット浮動小数2つのレジスタ（0で初期化）

    size_t i = 0;
    for (; i + 1 < vec.size(); i += 2) {
        __builtin_prefetch(&vec[i + 2], 0, 1);
        float64x2x2_t data = vld2q_f64(reinterpret_cast<const double*>(&vec[i]));

        float64x2_t realSquared = vmulq_f64(data.val[0], data.val[0]);
        float64x2_t imagSquared = vmulq_f64(data.val[1], data.val[1]);

        sum = vaddq_f64(sum, vaddq_f64(realSquared, imagSquared));
    }

    // NEONレジスタから部分和を取り出す
    double result[2];
    vst1q_f64(result, sum);

    // スカラー演算で奇数個目の要素を処理
    double scalarSum = result[0] + result[1];
    if (i < vec.size()) {
        scalarSum += std::pow(std::abs(vec[i]), 2);  // 最後の要素の絶対値の二乗を加算
    }

    return scalarSum;
#elif defined(__riscv) || defined(__riscv__)

#elif defined(__powerpc__) || defined(__PPC__)

#else
#endif
}
