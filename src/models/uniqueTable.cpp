#include "uniqueTable.hpp"

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable instance;
    return instance;
}

size_t UniqueTable::hash(size_t key) const {
    return key % ENTRY_COUNT;
}

void UniqueTable::insert(size_t hashKey, shared_ptr<QMDDNode> node) {
    unique_lock<shared_mutex> lock(tableMutex);
    size_t index = hash(hashKey);
    auto it = table.find(index);
    if (it != table.end()) {

        for (const auto& existingEntry : it->second) {
            if (existingEntry.key == hashKey) {
                return;
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
    for (const auto& entry : table) {
        size_t index = entry.first;
        const auto& entries = entry.second;

        cout << "Index: " << index << endl;
        cout << "Entries: " << endl;

        for (const auto& entry : entries) {
            cout << "  Key: " << entry.key << endl;
            if (entry.value) {
                const QMDDNode& node = *(entry.value);
                cout << "  Node: " << node << endl;
            } else {
                cout << "  Null node" << endl;
            }
        }
        cout << endl;
    }
    cout << "Total entries: " << table.size() << endl;
}
