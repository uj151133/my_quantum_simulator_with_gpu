#include "uniqueTable.hpp"

extern "C" {
    #include "../atomic/atomic.h"
    }

UniqueTable::UniqueTable() : table(tableSize) {
    for (auto& entry : table) entry.store(nullptr, memory_order_relaxed);
}

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable instance;
    return instance;
}

void UniqueTable::insert(long long hashKey, shared_ptr<QMDDNode> node) {
    long long idx = hash(hashKey);
    Entry* newEntry = new Entry(hashKey, node, nullptr);
    Entry* oldHead;
    while (true) {
        oldHead = table[idx].load(memory_order_acquire);
        // std::cout << "Debug: oldHead = " << oldHead << std::endl;
        // cout << "hashKey: " << hashKey << endl;
        if (oldHead == nullptr) {
            if (cas((void**)&table[idx], nullptr, newEntry)) break;
        } else {
            for (Entry* p = oldHead; p != nullptr; p = p->next) {
                if (p->key == hashKey && p->value == node) {
                    delete newEntry;
                    return;
                }
            }
            newEntry->next = oldHead;
            if (cas((void**)&table[idx], oldHead, newEntry)) break;
        }
        boost::this_fiber::yield();
    }
}

shared_ptr<QMDDNode> UniqueTable::find(long long hashKey) const {
    size_t idx = hash(hashKey);
    Entry* head = table[idx].load(memory_order_acquire);
    for (Entry* p = head; p != nullptr; p = p->next) {
        if (p->key == hashKey) {
            return p->value;
        }
    }
    return nullptr;
}

// 実装ファイル
long long UniqueTable::hash(long long key) const {
    return key & (tableSize - 1);
}

void UniqueTable::printAllEntries() const {
    int validEntries = 0;
    int invalidEntries = 0;
    for (size_t idx = 0; idx < tableSize; ++idx) {
        Entry* head = table[idx].load(memory_order_acquire);
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
    cout << "Total entries: (unknown in vector mode)" << endl;
    cout << "Table size: " << tableSize << endl;
    cout << "Valid entries: " << validEntries << endl;
    cout << "Invalid entries: " << invalidEntries << endl;
    cout << "Table bucket count: " << tableSize << endl;
}