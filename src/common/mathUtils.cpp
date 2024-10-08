#include "mathUtils.hpp"

QMDDEdge mathUtils::multiplication(const QMDDEdge& edge1, const QMDDEdge& edge2) {
    OperationCache& cache = OperationCache::getInstance();
    UniqueTable& table = UniqueTable::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(edge1, OperationType::MUL, edge2));
    cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        cout << "\033[1;35mCache miss!\033[0m" << endl;
        shared_ptr<QMDDNode> node1 = table.find(edge1.uniqueTableKey);
        shared_ptr<QMDDNode> node2 = table.find(edge2.uniqueTableKey);

        // edge1がターミナルノードである場合
        if (edge1.isTerminal) {
            QMDDEdge result = edge2;
            result.weight *= edge1.weight;
            return result;
        }
        // edge2がターミナルノードである場合
        if (edge2.isTerminal) {
            QMDDEdge result = edge1;
            result.weight *= edge2.weight;
            return result;
        }

        if (!node1 || !node2) {
            throw invalid_argument("\033[1;31mInvalid node pointer in QMDDEdge.\033[0m");
        }
        if  (node1->edges[0].size() != node2->edges.size()) {
            throw runtime_error("\033[1;31mNode edge sizes do not match for multiplication.\033[0m");
        }
        auto node1Copy = make_shared<QMDDNode>(*node1);
        auto node2Copy = make_shared<QMDDNode>(*node2);

        // 子ノードの重みを掛ける
        for (int i = 0; i < node1Copy->edges.size(); ++i) {
            for (int j = 0; j < node1Copy->edges[i].size(); ++j) {
                node1Copy->edges[i][j].weight *= edge1.weight;
            }
        }

        for (int i = 0; i < node2Copy->edges.size(); ++i) {
            for (int j = 0; j < node2Copy->edges[i].size(); ++j) {
                node2Copy->edges[i][j].weight *= edge2.weight;
            }
        }

        // 新しいエッジのベクトルを作成し、再帰的に子ノードを掛ける
        size_t l = node1Copy->edges.size();
        size_t m = node1Copy->edges[0].size();
        size_t n = node2Copy->edges[0].size();

        vector<vector<QMDDEdge>> newEdges(l, vector<QMDDEdge>(n, QMDDEdge(.0, nullptr)));

        for (size_t i = 0; i < l; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t k = 0; k < m; ++k) {
                    cout << node1->edges[i][k].weight << " * " << node2->edges[k][j].weight << endl;
                    newEdges[i][j] = mathUtils::addition(newEdges[i][j], mathUtils::multiplication(node1Copy->edges[i][k], node2Copy->edges[k][j]));
                    cout << "(" << i << ", " << j << "): " <<newEdges[i][j].weight << endl;
                }
            }
        }
        auto newNode = make_shared<QMDDNode>(newEdges);
        cache.insert(operationCacheKey, make_pair(1.0, calculation::generateUniqueTableKey(*newNode)));
        return QMDDEdge(1.0, newNode);
    }
}

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
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n1->edges[0].size(); j++){
                for (size_t k = 0; k < n0->edges[0].size(); k++) {
                    QMDDEdge p(e0Copy->weight * n0->edges[i][k].weight, table.find(n0->edges[i][k].uniqueTableKey));
                    QMDDEdge q(e1Copy->weight * n1->edges[k][j].weight, table.find(n1->edges[k][j].uniqueTableKey));
                    z[i][j] = add(z[i][j], mul(p, q));
                }
            }
        }
        auto newNode = make_shared<QMDDNode>(z);
        cache.insert(operationCacheKey, make_pair(1.0, calculation::generateUniqueTableKey(*newNode)));
        return QMDDEdge(1.0, newNode);
    }

}

QMDDEdge mathUtils::addition(const QMDDEdge& edge1, const QMDDEdge& edge2) {
    OperationCache& cache = OperationCache::getInstance();
    UniqueTable& table = UniqueTable::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(edge1, OperationType::ADD, edge2));
    cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        cout << "\033[1;35mCache miss!\033[0m" << endl;
        shared_ptr<QMDDNode> node1 = table.find(edge1.uniqueTableKey);
        shared_ptr<QMDDNode> node2 = table.find(edge2.uniqueTableKey);

        if (edge1.isTerminal) {
            QMDDEdge result = edge2;
            result.weight += edge1.weight;
            return result;
        }
        if (edge2.isTerminal) {
            QMDDEdge result = edge1;
            result.weight += edge2.weight;
            return result;
        }

        if (node1->edges.size() != node2->edges.size() || node1->edges[0].size() != node2->edges[0].size()) {
            throw runtime_error("\033[1;31mNode edge sizes do not match for addition.\033[0m");
        }
        if (!node1 || !node2) {
            throw invalid_argument("\033[1;31mInvalid node pointer in QMDDEdge.\033[0m");
        }

        // node1とnode2のコピーを作成
        auto node1Copy = make_shared<QMDDNode>(*node1);
        auto node2Copy = make_shared<QMDDNode>(*node2);

        for (size_t i = 0; i < node1Copy->edges.size(); ++i) {
            for (size_t j = 0; j < node1Copy->edges[i].size(); ++j) {
                node1Copy->edges[i][j].weight *= edge1.weight;
                node2Copy->edges[i][j].weight *= edge2.weight;
            }
        }

        vector<vector<QMDDEdge>> newEdges(node1Copy->edges.size(), vector<QMDDEdge>(node1Copy->edges[0].size()));
        for (size_t i = 0; i < newEdges.size(); ++i) {
            for (size_t j = 0; j < newEdges[i].size(); ++j) {
                newEdges[i][j] = mathUtils::addition(node1Copy->edges[i][j], node2Copy->edges[i][j]);
            }
        }

        auto newNode = make_shared<QMDDNode>(newEdges);
        complex<double> weight = 1.0;
        if (edge1.weight == .0 && edge2.weight == .0) {
            weight = .0;
        }
        cache.insert(operationCacheKey, make_pair(weight, calculation::generateUniqueTableKey(*newNode)));
        return QMDDEdge(weight, newNode);
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
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                QMDDEdge p(e0Copy->weight * n0->edges[i][j].weight, table.find(n0->edges[i][j].uniqueTableKey));
                QMDDEdge q(e1Copy->weight * n1->edges[i][j].weight, table.find(n1->edges[i][j].uniqueTableKey));
                z[i][j] = add(p, q);
            }
        }
        auto newNode = make_shared<QMDDNode>(z);
        cache.insert(operationCacheKey, make_pair(1.0, calculation::generateUniqueTableKey(*newNode)));
        return QMDDEdge(1.0, newNode);
    }
}

QMDDEdge mathUtils::kroneckerProduct(const QMDDEdge& edge1, const QMDDEdge& edge2) {
    OperationCache& cache = OperationCache::getInstance();
    UniqueTable& table = UniqueTable::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(edge1, OperationType::KRONECKER, edge2));
    cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        cout << "\033[1;36mCache hit!\033[0m" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        cout << "\033[1;35mCache miss!\033[0m" << endl;
        // 端点かどうかを確認
    if (edge1.isTerminal) {
        if (edge1.weight == .0) {
            // return edge1;
            // edge2のノードの深さを取得
            auto currentNode = table.find(edge2.uniqueTableKey);

            cout << "currentNode: " << edge2.uniqueTableKey << endl;
            QMDDEdge zeroEdge(.0, nullptr);

            while (currentNode && !currentNode->edges.empty()) {
                auto zeroNode = make_shared<QMDDNode>(vector<vector<QMDDEdge>>(currentNode->edges.size(), vector<QMDDEdge>(currentNode->edges[0].size(), zeroEdge)));
                zeroEdge = QMDDEdge(.0, zeroNode);
                currentNode = table.find(currentNode->edges[0][0].uniqueTableKey);
            }
            cout << "zeroEdge: " << zeroEdge.uniqueTableKey << endl;
            return zeroEdge;
        }
        if (edge1.weight == 1.0) {
            return edge2;
        }
        // それ以外のケース
        return QMDDEdge(edge1.weight * edge2.weight, edge2.uniqueTableKey);
    }

    // ノードへのポインタを取得
    shared_ptr<QMDDNode> node1 = table.find(edge1.uniqueTableKey);
    shared_ptr<QMDDNode> node2 = table.find(edge2.uniqueTableKey);

    auto node1Copy = make_shared<QMDDNode>(*node1);
    auto node2Copy = make_shared<QMDDNode>(*node2);
    vector<vector<QMDDEdge>> newEdges(node1Copy->edges.size(), vector<QMDDEdge>(node1Copy->edges[0].size()));
    for (size_t i = 0; i < newEdges.size(); ++i) {
        for (size_t j = 0; j < newEdges[i].size(); ++j) {
            newEdges[i][j] = mathUtils::kroneckerProduct(node1Copy->edges[i][j], edge2);
        }
    }
    auto newNode = make_shared<QMDDNode>(newEdges);

    cache.insert(operationCacheKey, make_pair(1.0, calculation::generateUniqueTableKey(*newNode)));
    return QMDDEdge(1.0, newNode);
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
        for (size_t i = 0; i < n0->edges.size(); i++) {
            for (size_t j = 0; j < n0->edges[i].size(); j++) {
                z[i][j] = kron(n0->edges[i][j], e1);
            }
        }
        auto newNode = make_shared<QMDDNode>(z);
        cache.insert(operationCacheKey, make_pair(1.0, calculation::generateUniqueTableKey(*newNode)));
        return QMDDEdge(1.0, newNode);
    }
}

QMDDEdge mathUtils::mulAny(QMDDEdge& e, int times) {
    if (times == 0) {
        return QMDDEdge(0.0, nullptr);
    }
    if (times == 1) {
        return e;
    }
    QMDDEdge result = e;
    for (int i = 1; i < times; ++i) {
        result = mathUtils::multiplication(result, e);
    }
    return result;
}
