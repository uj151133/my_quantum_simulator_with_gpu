#include "uniqueTable.hpp"

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable instance;
    return instance;
}

void UniqueTable::insert(size_t hashKey, shared_ptr<QMDDNode> node) {
    unique_lock<shared_mutex> lock(tableMutex);
    table[hashKey].push_back(node);
}

shared_ptr<QMDDNode> UniqueTable::find(size_t hashKey) const {
    shared_lock<shared_mutex> lock(tableMutex);
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
                cout << "  " << node << endl;
            } else {
                cout << "  Null node" << endl;
            }
        }
        cout << endl;
    }
    cout << "Total entries: " << table.size() << endl;
}
