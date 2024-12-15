#include "uniqueTable.hpp"

UniqueTable::UniqueTable() {
    // CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    tableSize = CONFIG.table.size;
    table.reserve(tableSize);
}

size_t UniqueTable::hash(size_t hashKey) const {
    return hashKey % tableSize;
}

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable instance;
    return instance;
}

void UniqueTable::insert(size_t hashKey, shared_ptr<QMDDNode> node) {
    unique_lock<shared_mutex> lock(tableMutex);
    size_t index = hash(hashKey);
    auto it = table.find(index);
    if (it != table.end()) {
        for (auto& existingEntry : it->second) {
            if (existingEntry.key == hashKey) {
                if (existingEntry.value == node) return;
                // else if (existingEntry.value.expired()) {
                //     existingEntry.value = weak_ptr<QMDDNode>(node);
                //     return;
                // }
            }
        }
    }
    table[index].push_back(Entry(hashKey, node));
    return;
}

shared_ptr<QMDDNode> UniqueTable::find(size_t hashKey) const {
    shared_lock<shared_mutex> lock(tableMutex);
    size_t index = hash(hashKey);
    auto it = table.find(index);
    if (it != table.end()) {
        for (const auto& entry : it->second) {
            if (entry.key == hashKey) {
                return entry.value;
            }
        }
    }
    return nullptr;
}

void UniqueTable::printAllEntries() const {
    for (const auto& item : table) {
        size_t index = item.first;
        const auto& entries = item.second;

        cout << "Index: " << index << endl;
        
        for (const auto& entry : entries) {
            cout << "  Key: " << entry.key << endl;
            cout << "  Nodes: " << endl;
            if (entry.value) {
                const QMDDNode& node = *entry.value;
                cout << "    " << node << endl;
            } else {
                cout << "    Null node" << endl;
            }
        }
        cout << endl;
    }
    cout << "Total entries: " << table.size() << endl;
    cout << "Table size: " << tableSize << endl;
    cout << "Table max bucket count: " << table.max_bucket_count() << endl;
    cout << "Table bucket count: " << table.bucket_count() << endl;
}
