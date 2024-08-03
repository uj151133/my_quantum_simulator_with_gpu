#include "uniqueTable.hpp"

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable instance;
    return instance;
}

void UniqueTable::insertNode(size_t hashKey, std::shared_ptr<QMDDNode> node) {
    table[hashKey].push_back(node);
}

std::shared_ptr<QMDDNode> UniqueTable::findNode(size_t hashKey, std::shared_ptr<QMDDNode> node) {
    auto it = table.find(hashKey);
    if (it != table.end()) {
        for (auto& existingNode : it->second) {
            if (*existingNode == *node) {
                return existingNode;
            }
        }
    }
    return nullptr;
}
