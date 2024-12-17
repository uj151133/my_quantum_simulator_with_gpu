#include "uniqueTable.hpp"

UniqueTable::UniqueTable() {
    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif
    
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
            if (entry.key == hashKey) {
                return entry.value.lock();
            }
        }
    }
    return nullptr;
}

void UniqueTable::printAllEntries() const {
    int validEntries = 0;
    int invalidEntries = 0;
    for (const auto& item : table) {
        size_t index = item.first;
        const auto& entries = item.second;

        cout << "Index: " << index << endl;
        
        for (const auto& entry : entries) {
            cout << "  Key: " << entry.key << endl;
            cout << "  Nodes: " << endl;
            if (!entry.value.expired()) {
                const QMDDNode& node = *entry.value.lock();
                cout << "    " << node << endl;
                validEntries++;
            } else {
                cout << "    Null node" << endl;
                invalidEntries++;
            }
        }
        cout << endl;
    }
    cout << "Total entries: " << table.size() << endl;
    cout << "Table size: " << tableSize << endl;
    cout << "Valid entries: " << validEntries << endl;
    cout << "Invalid entries: " << invalidEntries << endl;
    cout << "Table bucket count: " << table.bucket_count() << endl;
}
