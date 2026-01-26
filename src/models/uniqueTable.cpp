#include "uniqueTable.hpp"

extern "C" {
    #include "../atomic/atomic.h"
}

UniqueTable::UniqueTable() : tableSize_(PARAMETER.table.size), table_(this->tableSize_) {
    for (auto& entry : this->table_) entry.store(nullptr, memory_order_relaxed);
}

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable instance;
    return instance;
}

void UniqueTable::insert(int64_t hashKey, shared_ptr<QMDDNode> node) {
    int64_t idx = hash(hashKey);
    Entry* newEntry = new Entry(hashKey, weak_ptr<QMDDNode>(node), nullptr);
    Entry* oldHead;
    while (true) {
        oldHead = this->table_[idx].load(memory_order_acquire);
        for (Entry* p = oldHead; p != nullptr; p = p->next) {
            if (p->key == hashKey) {
                if (p->value.lock() == node) {
                    delete newEntry;
                    return;
                } else if (p->value.expired()) {
                    p->value = weak_ptr<QMDDNode>(node);
                    delete newEntry;
                    return;
                }
            }
        }
        newEntry->next = oldHead;
        if (cas((void**)&this->table_[idx], oldHead, newEntry)) break;
        boost::this_fiber::yield();
    }
}

shared_ptr<QMDDNode> UniqueTable::find(int64_t hashKey) const {
    size_t idx = hash(hashKey);
    Entry* head = this->table_[idx].load(memory_order_acquire);
    for (Entry* p = head; p != nullptr; p = p->next) {
        if (p->key == hashKey) {
            return p->value.lock();
        }
    }
    return nullptr;
}

int64_t UniqueTable::hash(int64_t key) const {
    return key & (this->tableSize_ - 1);
}

void UniqueTable::printAllEntries() const {
    int validEntries = 0;
    int invalidEntries = 0;
    for (size_t idx = 0; idx < this->tableSize_; ++idx) {
        Entry* head = this->table_[idx].load(memory_order_acquire);
        if (!head) continue;
        cout << "Index: " << idx << endl;
        for (Entry* p = head; p != nullptr; p = p->next) {
            cout << "  Key: " << p->key << endl;
            cout << "  Nodes: " << endl;
            if (p->value.lock()) {
                cout << "    " << *p->value.lock() << endl;
                validEntries++;
            } else {
                cout << "    Null node" << endl;
                invalidEntries++;
            }
        }
        cout << endl;
    }
    cout << "Total entries(unknown in vector mode): "  << validEntries + invalidEntries << endl;
    cout << "Table size: " << this->tableSize_ << endl;
    cout << "Valid entries: " << validEntries << endl;
    cout << "Invalid entries: " << invalidEntries << endl;
}

void UniqueTable::printNodeNum() const {
    int validEntries = 0;
    int invalidEntries = 0;
    for (size_t idx = 0; idx < this->tableSize_; ++idx) {
        Entry* head = this->table_[idx].load(memory_order_acquire);
        if (!head) continue;
        for (Entry* p = head; p != nullptr; p = p->next) {
            if (p->value.lock()) {
                validEntries++;
            } else {
                invalidEntries++;
            }
        }
    }
    cout << "Total entries(unknown in vector mode): "  << validEntries + invalidEntries << endl;
    cout << "Table size: " << this->tableSize_ << endl;
    cout << "Valid entries: " << validEntries << endl;
    cout << "Invalid entries: " << invalidEntries << endl;
    cout << "Table bucket count: " << this->tableSize_ << endl;
}

int UniqueTable::getTotalEntryCount() const {
    int totalEntries = 0;
    for (size_t idx = 0; idx < this->tableSize_; ++idx) {
        Entry* head = this->table_[idx].load(memory_order_acquire);
        for (Entry* p = head; p != nullptr; p = p->next) {
            totalEntries++;
        }
    }
    return totalEntries;
}