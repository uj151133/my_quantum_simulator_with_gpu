#include "mathUtils.hpp"

QMDDEdge mathUtils::mul(const QMDDEdge& e0, const QMDDEdge& e1) {
    // printf("thread = %d\n", omp_get_thread_num());
    OperationCache& cache = OperationCache::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::MUL, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        // cout << "\033[1;35mCache miss!\033[0m" << endl;

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
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n1->edges[0].size(); j++){
                for (size_t k = 0; k < n0->edges[0].size(); k++) {
                    QMDDEdge p(e0.weight * n0->edges[i][k].weight, n0->edges[i][k].uniqueTableKey);
                    QMDDEdge q(e1.weight * n1->edges[k][j].weight, n1->edges[k][j].uniqueTableKey);
                    z[i][j] += mathUtils::mul(p, q);
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
        QMDDEdge result;
        if (allWeightsAreZero) {
            result = QMDDEdge(.0, nullptr);
        } else {
            result = QMDDEdge(tmpWeight, make_shared<QMDDNode>(z));
        }
        cache.insert(operationCacheKey, make_pair(tmpWeight, result.uniqueTableKey));
        return result;
    }
}

QMDDEdge mathUtils::mulParallel(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::MUL, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        // cout << "\033[1;35mCache miss!\033[0m" << endl;

        QMDDEdge* e0Copy = const_cast<QMDDEdge*>(&e0);
        QMDDEdge* e1Copy = const_cast<QMDDEdge*>(&e1);
        if (e1Copy->isTerminal) {
            std::swap(e0Copy, e1Copy);
        }
        if (e0Copy->isTerminal) {
            if (e0Copy->weight == .0) {
                return *e0Copy;
            } else if (e0Copy->weight == 1.0){
                return *e1Copy;
            } else {
                return QMDDEdge(e0Copy->weight * e1Copy->weight, e1Copy->uniqueTableKey);
            }
        }

        shared_ptr<QMDDNode> n0 = e0.getStartNode();
        shared_ptr<QMDDNode> n1 = e1.getStartNode();

        vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size(), QMDDEdge(.0, nullptr)));
        complex<double> tmpWeight = .0;
        bool allWeightsAreZero = true;

        #pragma omp parallel for schedule(dynamic) collapse(2) default(shared)
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n1->edges[0].size(); j++){
                QMDDEdge r(.0, nullptr);
                for (size_t k = 0; k < n0->edges[0].size(); k++) {
                    QMDDEdge p = QMDDEdge(e0Copy->weight * n0->edges[i][k].weight, n0->edges[i][k].uniqueTableKey);
                    QMDDEdge q = QMDDEdge(e1Copy->weight * n1->edges[k][j].weight, n1->edges[k][j].uniqueTableKey);
                    r = mathUtils::add(r, mathUtils::mul(p, q));
                }
                z[i][j] = r;
            }
        }

        for (size_t i = 0; i < z.size(); i++) {
            for (size_t j = 0; j < z[i].size(); j++) {
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
}

QMDDEdge mathUtils::mulForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::MUL, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        // cout << "\033[1;35mCache miss!\033[0m" << endl;

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
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n1->edges[0].size(); j++){
                for (size_t k = 0; k < n0->edges[0].size(); k++) {
                    QMDDEdge p(e0.weight * n0->edges[i][k].weight, n0->edges[i][k].uniqueTableKey);
                    QMDDEdge q(e1.weight * n1->edges[k][j].weight, n1->edges[k][j].uniqueTableKey);
                    z[i][j] = mathUtils::add(z[i][j], mathUtils::mulForDiagonal(p, q));
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
        QMDDEdge result;
        if (allWeightsAreZero) {
            result = QMDDEdge(.0, nullptr);
        } else {
            result = QMDDEdge(tmpWeight, make_shared<QMDDNode>(z));
        }
        cache.insert(operationCacheKey, make_pair(tmpWeight, result.uniqueTableKey));
        return result;
    }
}

QMDDEdge mathUtils::add(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::ADD, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        // cout << "\033[1;35mCache miss!\033[0m" << endl;

        QMDDEdge* e0Copy = const_cast<QMDDEdge*>(&e0);
        QMDDEdge* e1Copy = const_cast<QMDDEdge*>(&e1);
        if (e1Copy->isTerminal) {
            std::swap(e0Copy, e1Copy);
        }
        if (e0Copy->isTerminal) {
            if (e0Copy->weight == .0) {
                return *e1Copy;
            } else if (e1Copy->isTerminal) {
                return QMDDEdge(e0Copy->weight + e1Copy->weight, nullptr);
            }
        }
        shared_ptr<QMDDNode> n0 = e0.getStartNode();
        shared_ptr<QMDDNode> n1 = e1.getStartNode();
        bool allWeightsAreZero = true;
        vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n0->edges[0].size()));
        complex<double> tmpWeight = .0;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                QMDDEdge p(e0Copy->weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
                QMDDEdge q(e1Copy->weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
                z[i][j] = mathUtils::add(p, q);

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
}

QMDDEdge mathUtils::addParallel(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::ADD, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        // cout << "\033[1;35mCache miss!\033[0m" << endl;
        QMDDEdge* e0Copy = const_cast<QMDDEdge*>(&e0);
        QMDDEdge* e1Copy = const_cast<QMDDEdge*>(&e1);

        if (e1Copy->isTerminal) {
            std::swap(e0Copy, e1Copy);
        }
        if (e0Copy->isTerminal) {
            if (e0Copy->weight == .0) {
                return *e1Copy;
            } else if (e1Copy->isTerminal) {
                return QMDDEdge(e0Copy->weight + e1Copy->weight, nullptr);
            }
        }
        shared_ptr<QMDDNode> n0 = e0.getStartNode();
        shared_ptr<QMDDNode> n1 = e1.getStartNode();
        bool allWeightsAreZero = true;
        vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n0->edges[0].size()));
        complex<double> tmpWeight = .0;

        #pragma omp parallel for schedule(dynamic) default(shared)
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                QMDDEdge p(e0Copy->weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
                QMDDEdge q(e1Copy->weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
                z[i][j] = mathUtils::add(p, q);
            }
        }

        for (size_t i = 0; i < z.size(); i++) {
            for (size_t j = 0; j < z[i].size(); j++) {
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
}

QMDDEdge mathUtils::addForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::ADD, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        // cout << "\033[1;35mCache miss!\033[0m" << endl;
        
        QMDDEdge* e0Copy = const_cast<QMDDEdge*>(&e0);
        QMDDEdge* e1Copy = const_cast<QMDDEdge*>(&e1);
        if (e1Copy->isTerminal) {
            std::swap(e0Copy, e1Copy);
        }
        if (e0Copy->isTerminal) {
            if (e0Copy->weight == .0) {
                return *e1Copy;
            } else if (e1Copy->isTerminal) {
                return QMDDEdge(e0Copy->weight + e1Copy->weight, nullptr);
            }
        }
        shared_ptr<QMDDNode> n0 = e0.getStartNode();
        shared_ptr<QMDDNode> n1 = e1.getStartNode();
        bool allWeightsAreZero = true;
        vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n0->edges[0].size()));
        complex<double> tmpWeight = .0;
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                QMDDEdge p(e0Copy->weight * n0->edges[i][j].weight, n0->edges[i][j].uniqueTableKey);
                QMDDEdge q(e1Copy->weight * n1->edges[i][j].weight, n1->edges[i][j].uniqueTableKey);
                z[i][j] = mathUtils::addForDiagonal(p, q);

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
}

QMDDEdge mathUtils::kron(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::KRONECKER, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
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

        for (size_t i = 0; i < z.size(); i++) {
            for (size_t j = 0; j < z[i].size(); j++) {
                z[i][j] = mathUtils::kron(n0->edges[i][j], e1);
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
            result = QMDDEdge(e0.weight * tmpWeight, make_shared<QMDDNode>(z));
        }
        cache.insert(operationCacheKey, make_pair(result.weight, result.uniqueTableKey));
        return result;
    }
}

QMDDEdge mathUtils::kronParallel(const QMDDEdge& e0, const QMDDEdge& e1) {
    // cout << "マルチスレッド処理: スレッド" << ": " << omp_get_thread_num() << endl;
    OperationCache& cache = OperationCache::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::KRONECKER, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
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

        #pragma omp parallel for s_chedule(dynamic) default(shared)
        for (size_t i = 0; i < z.size(); i++) {
            for (size_t j = 0; j < z[i].size(); j++) {
                z[i][j] = mathUtils::kron(n0->edges[i][j], e1);
            }
        }

        for (size_t i = 0; i < z.size(); i++) {
            for (size_t j = 0; j < z[i].size(); j++) {
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
            result = QMDDEdge(e0.weight * tmpWeight, make_shared<QMDDNode>(z));
        }
        cache.insert(operationCacheKey, make_pair(result.weight, result.uniqueTableKey));
        return result;
    }
}

QMDDEdge mathUtils::kronForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::KRONECKER, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        // cout << "\033[1;35mCache miss!\033[0m" << endl;

        QMDDEdge* e0Copy = const_cast<QMDDEdge*>(&e0);
        QMDDEdge* e1Copy = const_cast<QMDDEdge*>(&e1);
        if (e0Copy->isTerminal) {
            if (e0Copy->weight == .0) {
                return *e0Copy;
            }else if (e0Copy->weight == 1.0) {
                return *e1Copy;
            } else {
                return QMDDEdge(e0Copy->weight * e1Copy->weight, e1Copy->uniqueTableKey);
            }
        }
        shared_ptr<QMDDNode> n0 = e0.getStartNode();
        shared_ptr<QMDDNode> n1 = e1.getStartNode();
        vector<vector<QMDDEdge>> z(2, vector<QMDDEdge>(2, QMDDEdge(.0, nullptr)));
        complex<double> tmpWeight = .0;
        bool allWeightsAreZero = true;
        // #pragma omp parallel for shared(z) num_threads(2) schedule(auto)
        for (size_t n = 0; n < 2; n++) {
            z[n][n] = mathUtils::kronForDiagonal(n0->edges[n][n], e1);
            if (z[n][n].weight != .0) {
                allWeightsAreZero = false;
                if (tmpWeight == .0) {
                    tmpWeight = z[n][n].weight;
                    z[n][n].weight = 1.0;
                }else if (tmpWeight != .0) {
                    z[n][n].weight /= tmpWeight;
                } else {
                    cout << "⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️" << endl;
                }
            }
        }
        QMDDEdge result;
        if (allWeightsAreZero) {
            result = QMDDEdge(.0, nullptr);
        } else {
            result = QMDDEdge(e0Copy->weight * tmpWeight, make_shared<QMDDNode>(z));
        }
        cache.insert(operationCacheKey, make_pair(result.weight, result.uniqueTableKey));
        return result;
    }
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

double mathUtils::sumOfSquares(const vector<complex<double>>& vec) {
    return accumulate(vec.begin(), vec.end(), 0.0, [](double sum, const complex<double>& val) {
        return sum + std::pow(abs(val), 2);
    });
}

