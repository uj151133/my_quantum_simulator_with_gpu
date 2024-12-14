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

        for (auto& existingEntry : it->second) {
            if (existingEntry.key == hashKey) {
                if (existingEntry.value.lock() == node) return;
                else if (existingEntry.value.expired()) {
                    existingEntry.value = weak_ptr<QMDDNode>(node);
                    return;
                }
            }
        }
    }
    table[index].push_back(Entry(hashKey, weak_ptr<QMDDNode>(node)));
    return;
}

shared_ptr<QMDDNode> UniqueTable::find(size_t hashKey) const {
    shared_lock<shared_mutex> lock(tableMutex);
    size_t index = hash(hashKey);
    auto it = table.find(index);
    if (it != table.end()) {
        for (const auto& entry : it->second) {
            if (entry.key == hashKey && !entry.value.expired()) {
                return entry.value.lock();
            }
        }
    }
    return nullptr;
}

void UniqueTable::printAllEntries() const {
    size_t validEntries = 0;
    for (const auto& entry : table) {
        size_t index = entry.first;
        const auto& entries = entry.second;

        cout << "Index: " << index << endl;
        cout << "Entries: " << endl;

        for (const auto& entry : entries) {
            cout << "  Key: " << entry.key << endl;
            if (!entry.value.expired()) {
                const QMDDNode& node = *(entry.value.lock());
                cout << "  Node: " << node << endl;
                validEntries++;
            } else {
                cout << "  Node expired!" << endl;
            }
        }
        cout << endl;
    }
    cout << "Total entries: " << table.size() << endl;
    cout << "Valid (non-expired) entries: " << validEntries << endl;
}
