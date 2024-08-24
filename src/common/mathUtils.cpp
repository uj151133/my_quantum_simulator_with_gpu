#include "mathUtils.hpp"

using namespace std;

QMDDEdge mathUtils::multiplication(const QMDDEdge& edge1, const QMDDEdge& edge2) {
    OperationCache& cache = OperationCache::getInstance();
    UniqueTable& table = UniqueTable::getInstance();
    size_t operationCacheKey = calculation::generateOperationCacheKey(make_tuple(edge1, OperationType::MUL, edge2));
    cout << "Operation cache key: " << operationCacheKey << endl;
    auto existingAnswer = cache.find(operationCacheKey);
    if (existingAnswer != OperationResult{.0, 0}) {
        cout << "Cache hit!" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        cout << "Cache miss!" << endl;
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
            throw invalid_argument("Invalid node pointer in QMDDEdge.");
        }


        // 子ノードの重みを掛ける
        for (int i = 0; i < node1->edges.size(); ++i) {
            for (int j = 0; j < node1->edges[i].size(); ++j) {
                node1->edges[i][j].weight *= edge1.weight;
            }
        }

        for (int i = 0; i < node2->edges.size(); ++i) {
            for (int j = 0; j < node2->edges[i].size(); ++j) {
                node2->edges[i][j].weight *= edge2.weight;
            }
        }

        // 新しいエッジのベクトルを作成し、再帰的に子ノードを掛ける
        size_t n = node1->edges.size();
        size_t m = node1->edges[0].size();
        vector<vector<QMDDEdge>> newEdges(n, vector<QMDDEdge>(m, QMDDEdge(0.0, nullptr)));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                for (size_t k = 0; k < m; ++k) {
                    newEdges[i][j] = mathUtils::addition(newEdges[i][j], mathUtils::multiplication(node1->edges[i][k], node2->edges[k][j]));
                }
            }
        }
        // vector<QMDDEdge> newEdge(node1->edges.size());
        // newEdge[0] = mathUtils::addition(mathUtils::addition(node1->edges[0], node2->edges[0]), mathUtils::addition(node1->edges[1], node2->edges[2]));
        // newEdge[1] = mathUtils::addition(mathUtils::addition(node1->edges[0], node2->edges[1]), mathUtils::addition(node1->edges[1], node2->edges[3]));
        // newEdge[2] = mathUtils::addition(mathUtils::addition(node1->edges[2], node2->edges[0]), mathUtils::addition(node1->edges[3], node2->edges[2]));
        // newEdge[3] = mathUtils::addition(mathUtils::addition(node1->edges[2], node2->edges[1]), mathUtils::addition(node1->edges[3], node2->edges[3]));

        // 新しいノードを作成し、キャッシュに結果を保存
        auto newNode = make_shared<QMDDNode>(newEdges);
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
        cout << "Cache hit!" << endl;
        return QMDDEdge(existingAnswer.first, existingAnswer.second);
    }
    else {
        cout << "Cache miss!" << endl;
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
            throw std::runtime_error("Node edge sizes do not match.");
        }
        if (!node1 || !node2) {
            throw invalid_argument("Invalid node pointer in QMDDEdge.");
        }

        for (size_t i = 0; i < node1->edges.size(); ++i) {
            for (size_t j = 0; j < node1->edges[i].size(); ++j) {
                node1->edges[i][j].weight *= edge1.weight;
                node2->edges[i][j].weight *= edge2.weight;
            }
        }

        vector<vector<QMDDEdge>> newEdges(node1->edges.size(), vector<QMDDEdge>(node1->edges[0].size()));
        for (size_t i = 0; i < newEdges.size(); ++i) {
            for (size_t j = 0; j < newEdges[i].size(); ++j) {
                newEdges[i][j] = mathUtils::addition(node1->edges[i][j], node2->edges[i][j]);
            }
        }

        auto newNode = make_shared<QMDDNode>(newEdges);
        cache.insert(operationCacheKey, make_pair(1.0, calculation::generateUniqueTableKey(*newNode)));
        return QMDDEdge(1.0, newNode);
    }
}