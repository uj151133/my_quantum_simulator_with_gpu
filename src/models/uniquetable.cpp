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
void UniqueTable::printAllEntries() const {
    for (const auto& entry : table) {
        size_t key = entry.first;
        const auto& nodes = entry.second;

        std::cout << "Key: " << key << std::endl;
        std::cout << "Nodes: " << std::endl;

        for (const auto& nodePtr : nodes) {
            if (nodePtr) {
                const QMDDNode& node = *nodePtr;
                std::cout << "  Node with " << node.edges.size() << " edges." << std::endl;
                for (size_t i = 0; i < node.edges.size(); ++i) {
                    const QMDDEdge& edge = node.edges[i];
                    std::cout << "    Edge " << i << ": weight = " << edge.weight << ", isTerminal = " << edge.isTerminal << std::endl;
                }
            } else {
                std::cout << "  Null node" << std::endl;
            }
        }
        std::cout << std::endl;
    }
}
