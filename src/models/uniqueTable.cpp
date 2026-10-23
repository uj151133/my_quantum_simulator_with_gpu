#include "uniqueTable.hpp"

UniqueTable::UniqueTable() : tableSize_(PARAMETER.table.size / 2), table_(this->tableSize_) {
    for (auto& entry : this->table_) entry.store(nullptr, memory_order_relaxed);
}

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable* instance = new UniqueTable();
    return *instance;
}

QMDDNode* UniqueTable::insert(int64_t hashKey, QMDDNode* node) {
    const int64_t idx = this->hash(hashKey);
    std::unique_lock<SpinLock> lk(this->locks_[this->shard(idx)]);
    QMDDNode* head = this->table_[idx].load(memory_order_acquire);
    for (QMDDNode* p = head; p != nullptr; p = p->next) {
        if (p == node) { p->refCount.fetch_add(1, memory_order_relaxed); return node; } 
        if (p->refCount.load(memory_order_acquire) == 0) continue;
        if (p->key == hashKey && *p == *node) { p->refCount.fetch_add(1, memory_order_relaxed); return p; }
    }
    node->key = hashKey;
    node->next = head;
    node->refCount.fetch_add(1, memory_order_relaxed);
    this->table_[idx].store(node, memory_order_release);
    return node;
}

QMDDNode* UniqueTable::find(int64_t hashKey) {
    const int64_t idx = this->hash(hashKey);
    std::unique_lock<SpinLock> lk(this->locks_[this->shard(idx)]);
    for (QMDDNode* p = this->table_[idx].load(memory_order_acquire); p != nullptr; p = p->next) {
        if (p->refCount.load(memory_order_acquire) == 0) continue; 
        if (p->key == hashKey) { p->refCount.fetch_add(1, memory_order_relaxed); return p; }
    }
    return nullptr;
}

void UniqueTable::release(QMDDNode* node) {
    const int64_t idx = this->hash(node->key);
    {
        std::unique_lock<SpinLock> lk(this->locks_[this->shard(idx)]);
        if (node->refCount.load(memory_order_acquire) != 0) return;
        QMDDNode* prev = nullptr;
        for (QMDDNode* p = this->table_[idx].load(memory_order_acquire); p != nullptr; prev = p, p = p->next) {
            if (p == node) {
                if (prev) prev->next = p->next; else this->table_[idx].store(p->next, memory_order_release);
                break;
            }
        }
    }
    delete node;
}

int64_t UniqueTable::hash(int64_t key) const {
    return key & (this->tableSize_ - 1);
}

void UniqueTable::printAllEntries() const {
    int entries = 0;
    for (int64_t idx = 0; idx < this->tableSize_; ++idx) {
        std::unique_lock<SpinLock> lk(this->locks_[this->shard(idx)]);
        QMDDNode* head = this->table_[idx].load(memory_order_acquire);
        if (!head) continue;
        cout << "Index: " << idx << endl;
        for (QMDDNode* p = head; p != nullptr; p = p->next) {
            cout << "  Key: " << p->key << " refCount: " << p->refCount.load() << endl;
            cout << "    " << *p << endl;
            entries++;
        }
    }
    cout << "Total entries: " << entries << endl;
    cout << "Table size: " << this->tableSize_ << endl;
}

void UniqueTable::printNodeNum() const {
    cout << "Total entries: " << this->getTotalEntryCount() << endl;
    cout << "Table size: " << this->tableSize_ << endl;
}

int UniqueTable::getTotalEntryCount() const {
    int total = 0;
    for (int64_t idx = 0; idx < this->tableSize_; ++idx) {
        std::unique_lock<SpinLock> lk(this->locks_[this->shard(idx)]);
        for (QMDDNode* p = this->table_[idx].load(memory_order_acquire); p != nullptr; p = p->next) total++;
    }
    return total;
}
