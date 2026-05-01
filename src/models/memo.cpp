#include "Memo.hpp"

extern "C" {
    #include "../atomic/atomic.h"
}

Memo::Memo()
    : tableSize_(PARAMETER.table.size / 2), table_(this->tableSize_) {
    for (auto& entry : this->table_) entry.store(nullptr, memory_order_relaxed);
}

Memo& Memo::getInstance() {
    static Memo instance;
    return instance;
}

void Memo::insert(int64_t hashKey, const shared_ptr<SVLeaf>& leaf) {
    int64_t idx = hash(hashKey);
    SVEntry* newEntry = new SVEntry(hashKey, weak_ptr<SVLeaf>(leaf), nullptr);

    SVEntry* oldHead;
    while (true) {
        oldHead = this->table_[idx].load(memory_order_acquire);

        for (SVEntry* p = oldHead; p != nullptr; p = p->next) {
            if (p->key == hashKey) {
                if (p->value.expired()) {
                    p->value = leaf;
                }
                delete newEntry;
                return;
            }
        }

        newEntry->next = oldHead;
        if (cas((void**)&this->table_[idx], oldHead, newEntry)) break;
        boost::this_fiber::yield();
    }
}

shared_ptr<SVLeaf> Memo::find(int64_t hashKey) const {
    size_t idx = hash(hashKey);
    SVEntry* head = this->table_[idx].load(memory_order_acquire);

    for (SVEntry* p = head; p != nullptr; p = p->next) {
        if (p->key == hashKey) {
            return p->value.lock().get();
        }
    }
    return nullptr;
}

int64_t Memo::hash(int64_t key) const {
    return key & (this->tableSize_ - 1);
}

void Memo::printAllEntries() const {
    int validEntries = 0;
    int invalidEntries = 0;

    for (size_t idx = 0; idx < this->tableSize_; ++idx) {
        SVEntry* head = this->table_[idx].load(memory_order_acquire);
        if (!head) continue;

        cout << "Index: " << idx << endl;
        for (SVEntry* p = head; p != nullptr; p = p->next) {
            cout << "  Key: " << p->key << endl;
            if (auto sp = p->value.lock()) {
                cout << "  SVLeaf valid(size=" << sp->size << ", sourceKey=" << sp->sourceKey << ")" << endl;
                validEntries++;
            } else {
                cout << "  Null leaf" << endl;
                invalidEntries++;
            }
        }
        cout << endl;
    }

    cout << "Total entries: " << validEntries + invalidEntries << endl;
    cout << "Table size: " << this->tableSize_ << endl;
    cout << "Valid entries: " << validEntries << endl;
    cout << "Invalid entries: " << invalidEntries << endl;
}

int Memo::getTotalEntryCount() const {
    int totalEntries = 0;
    for (size_t idx = 0; idx < this->tableSize_; ++idx) {
        SVEntry* head = this->table_[idx].load(memory_order_acquire);
        for (SVEntry* p = head; p != nullptr; p = p->next) {
            totalEntries++;
        }
    }
    return totalEntries;
}