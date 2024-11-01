#include "mathUtils.hpp"

QMDDEdge mathUtils::mul(const QMDDEdge& e0, const QMDDEdge& e1) {
    OperationCache& cache = OperationCache::getInstance();
    UniqueTable& table = UniqueTable::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::MUL, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        // cout << "\033[1;35mCache miss!\033[0m" << endl;
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
                return *e1Copy;
            } else {
                return QMDDEdge(e0Copy->weight * e1Copy->weight, n1);
            }
        }

        vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size(), QMDDEdge(.0, nullptr)));
        complex<double> tmpWeight = .0;
        bool allWeightsAreZero = true;
        boost::fibers::mutex mtx;
        boost::fibers::condition_variable cv;
        size_t remaining = n0->edges.size() * n1->edges[0].size();
        queue<pair<size_t, size_t>> taskQueue;

        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n1->edges[0].size(); j++) {
                taskQueue.push(make_pair(i, j));
            }
        }

        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n1->edges[0].size(); j++) {
                boost::fibers::fiber([&, i, j] {
                    for (size_t k = 0; k < n0->edges[0].size(); k++) {
                        QMDDEdge p(e0Copy->weight * n0->edges[i][k].weight, table.find(n0->edges[i][k].uniqueTableKey));
                        QMDDEdge q(e1Copy->weight * n1->edges[k][j].weight, table.find(n1->edges[k][j].uniqueTableKey));
                        {
                            std::unique_lock<boost::fibers::mutex> lock(mtx);
                            z[i][j] = mathUtils::add(z[i][j], mathUtils::mul(p, q));
                        }
                    }

                    {
                        std::unique_lock<boost::fibers::mutex> lock(mtx);
                        cv.wait(lock, [&] { return taskQueue.front() == make_pair(i, j); });

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

                        taskQueue.pop();
                        cv.notify_all();
                    }

                    {
                        std::unique_lock<boost::fibers::mutex> lock(mtx);
                        if (--remaining == 0) {
                            cv.notify_all();
                        }
                    }
                }).detach();
            }
        }

        {
            std::unique_lock<boost::fibers::mutex> lock(mtx);
            cv.wait(lock, [&] { return remaining == 0; });
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
    UniqueTable& table = UniqueTable::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::ADD, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        // cout << "\033[1;35mCache miss!\033[0m" << endl;
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
        bool allWeightsAreZero = true;
        vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n0->edges[0].size()));
        complex<double> tmpWeight = .0;
        boost::fibers::mutex mtx;
        boost::fibers::condition_variable cv;
        size_t remaining = n0->edges.size() * n0->edges[0].size();
        queue<pair<size_t, size_t>> taskQueue;

        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                taskQueue.push(make_pair(i, j));
            }
        }

        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                boost::fibers::fiber([&, i, j] {
                    QMDDEdge p(e0Copy->weight * n0->edges[i][j].weight, table.find(n0->edges[i][j].uniqueTableKey));
                    QMDDEdge q(e1Copy->weight * n1->edges[i][j].weight, table.find(n1->edges[i][j].uniqueTableKey));
                    z[i][j] = mathUtils::add(p, q);

                    {
                        std::unique_lock<boost::fibers::mutex> lock(mtx);
                        cv.wait(lock, [&] { return taskQueue.front() == make_pair(i, j); });

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

                        taskQueue.pop();
                        cv.notify_all();
                    }

                    {
                        std::unique_lock<boost::fibers::mutex> lock(mtx);
                        if (--remaining == 0) {
                            cv.notify_all();
                        }
                    }
                }).detach();
            }
        }
        {
            std::unique_lock<boost::fibers::mutex> lock(mtx);
            cv.wait(lock, [&] { return remaining == 0; });
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
    UniqueTable& table = UniqueTable::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(e0, OperationType::KRONECKER, e1));
    // cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        // cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        // cout << "\033[1;35mCache miss!\033[0m" << endl;
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
        vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size()));
        complex<double> tmpWeight = .0;
        bool allWeightsAreZero = true;
        boost::fibers::mutex mtx;
        boost::fibers::condition_variable cv;
        size_t remaining = n0->edges.size() * n0->edges[0].size();
        queue<pair<size_t, size_t>> taskQueue;

        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                taskQueue.push(make_pair(i, j));
            }
        }
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                boost::fibers::fiber([&, i, j] {
                    z[i][j] = mathUtils::kron(n0->edges[i][j], e1);

                    {
                        std::unique_lock<boost::fibers::mutex> lock(mtx);
                        cv.wait(lock, [&] { return taskQueue.front() == make_pair(i, j); });

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

                        taskQueue.pop();
                        cv.notify_all();
                    }

                    {
                        std::unique_lock<boost::fibers::mutex> lock(mtx);
                        if (--remaining == 0) {
                            cv.notify_all();
                        }
                    }
                }).detach();
            }
        }

        {
            std::unique_lock<boost::fibers::mutex> lock(mtx);
            cv.wait(lock, [&] { return remaining == 0; });
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

