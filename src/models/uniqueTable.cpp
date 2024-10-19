#include "uniqueTable.hpp"

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable instance;
    return instance;
}

void UniqueTable::insert(size_t hashKey, shared_ptr<QMDDNode> node) {
    lock_guard<mutex> lock(tableMutex);  // キャッシュ全体の操作を保護
    // auto it = table.find(hashKey);
    // if (it != table.end()) {
        // auto old_vector = it->second;  // 現在のベクトルを取得
        // auto new_vector = old_vector;  // 新しいベクトルを作成（コピー）
        // new_vector.push_back(node);    // 新しいノードを追加
        // compare_and_swap(it->second, it->second, new_vector);  // CASでベクトルを更新
    // } else {
        table[hashKey].push_back(node);  // 新規挿入
    // }
}

shared_ptr<QMDDNode> UniqueTable::find(size_t hashKey) const {
        auto it = table.find(hashKey);
        if (it != table.end()) {
            return it->second[0];
        }
        return nullptr;
}

void UniqueTable::printAllEntries() const {
    for (const auto& entry : table) {
        size_t key = entry.first;
        const auto& nodes = entry.second;

        cout << "Key: " << key << endl;
        cout << "Nodes: " << endl;

        for (const auto& nodePtr : nodes) {
            if (nodePtr) {
                const QMDDNode& node = *nodePtr;
                cout << "  Node with " << node.edges.size() << " rows of children." << endl;
                for (size_t i = 0; i < node.edges.size(); ++i) {
                    for (size_t j = 0; j < node.edges[i].size(); ++j) {
                        const QMDDEdge& edge = node.edges[i][j];
                        cout << "    Edge (" << i << ", " << j << "): weight = " << edge.weight << ", Key = " << edge.uniqueTableKey << ", isTerminal = " << edge.isTerminal << endl;
                    }
                }
            } else {
                cout << "  Null node" << endl;
            }
        }
        cout << endl;
    }
}
