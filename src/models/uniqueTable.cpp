#include "uniqueTable.hpp"

extern "C" {
    #include "../atomic/atomic.h"
    }

UniqueTable::UniqueTable() : table(tableSize) {
    for (auto& entry : this->table) entry.store(nullptr, memory_order_relaxed);
}

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable instance;
    return instance;
}

void UniqueTable::insert(int64_t hashKey, shared_ptr<QMDDNode> node) {
    int64_t idx = hash(hashKey);
    Entry* newEntry = new Entry(hashKey, node, nullptr);
    Entry* oldHead;
    while (true) {
        oldHead = this->table[idx].load(memory_order_acquire);
        for (Entry* p = oldHead; p != nullptr; p = p->next) {
            if (p == nullptr) break;
            if (p->key == hashKey) {
                delete newEntry;
                return;
            }
        }
        newEntry->next = oldHead;
        if (cas((void**)&this->table[idx], oldHead, newEntry)) break;
        boost::this_fiber::yield();
    }
}

shared_ptr<QMDDNode> UniqueTable::find(int64_t hashKey) const {
    size_t idx = hash(hashKey);
    Entry* head = this->table[idx].load(memory_order_acquire);
    for (Entry* p = head; p != nullptr; p = p->next) {
        if (p == nullptr) break;
        if (p->key == hashKey) {
            return p->value;
        }
    }
    return nullptr;
}

// 実装ファイル
int64_t UniqueTable::hash(int64_t key) const {
    return key & (this->tableSize - 1);
}

void UniqueTable::printAllEntries() const {
    int validEntries = 0;
    int invalidEntries = 0;
    for (size_t idx = 0; idx < this->tableSize; ++idx) {
        Entry* head = this->table[idx].load(memory_order_acquire);
        if (!head) continue;
        cout << "Index: " << idx << endl;
        for (Entry* p = head; p != nullptr; p = p->next) {
            cout << "  Key: " << p->key << endl;
            cout << "  Nodes: " << endl;
            if (p->value) {
                cout << "    " << *p->value << endl;
                validEntries++;
            } else {
                cout << "    Null node" << endl;
                invalidEntries++;
            }
        }
        cout << endl;
    }
    cout << "Total entries(unknown in vector mode): "  << validEntries + invalidEntries << endl;
    cout << "Table size: " << this->tableSize << endl;
    cout << "Valid entries: " << validEntries << endl;
    cout << "Invalid entries: " << invalidEntries << endl;
    cout << "Table bucket count: " << this->tableSize << endl;
}

void UniqueTable::printNodeNum() const {
    int validEntries = 0;
    int invalidEntries = 0;
    for (size_t idx = 0; idx < this->tableSize; ++idx) {
        Entry* head = this->table[idx].load(memory_order_acquire);
        if (!head) continue;
        for (Entry* p = head; p != nullptr; p = p->next) {
            if (p->value) {
                validEntries++;
            } else {
                invalidEntries++;
            }
        }
    }
    cout << "Total entries(unknown in vector mode): "  << validEntries + invalidEntries << endl;
    cout << "Table size: " << this->tableSize << endl;
    cout << "Valid entries: " << validEntries << endl;
    cout << "Invalid entries: " << invalidEntries << endl;
    cout << "Table bucket count: " << this->tableSize << endl;
}

int UniqueTable::getTotalEntryCount() const {
    int totalEntries = 0;
    for (size_t idx = 0; idx < this->tableSize; ++idx) {
        Entry* head = this->table[idx].load(memory_order_acquire);
        for (Entry* p = head; p != nullptr; p = p->next) {
            totalEntries++;
        }
    }
    return totalEntries;
}