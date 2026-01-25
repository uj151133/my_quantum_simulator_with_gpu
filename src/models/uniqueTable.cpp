#include "uniqueTable.hpp"

extern "C" {
    #include "../atomic/atomic.h"
}

UniqueTable::UniqueTable(){
    PARAMETER.load();
    this->tableSize_ = PARAMETER.table.size;
    if (this->tableSize_ <= 0) this->tableSize_ = 1;
    // this->table_(static_cast<size_t>(tableSize_));
    this->table_ = std::vector<std::atomic<Entry*>>(this->tableSize_);
    for (auto& entry : this->table_) entry.store(nullptr, memory_order_relaxed);
}

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable instance;
    return instance;
}

void UniqueTable::insert(int64_t hashKey, shared_ptr<QMDDNode> node) {
    // cout << "UniqueTable::insert called with hashKey: " << hashKey << endl;
    int64_t idx = hash(hashKey);
    Entry* newEntry = new Entry(hashKey, weak_ptr<QMDDNode>(node), nullptr);
    Entry* head;
    int loop_count = 0;
    while (true) {
        head = this->table_[idx].load(memory_order_acquire);

        // while (head != nullptr) {
        //     // expired 判定は bool を最優先
        //     if (!head->waste.load(memory_order_acquire)) {
        //         // value が死んでたら expired にする
        //         if (head->value.lock() == nullptr) {
        //             head->waste.store(true, memory_order_release);
        //         } else {
        //             break; // 生きてる head
        //         }
        //     }

        //     // expired head を物理削除
        //     Entry* next = head->next;
        //     if (cas((void**)&this->table_[idx], head, next)) {
        //         this->throwAway(head);
        //         head = next;
        //     }
        // }

        for (Entry* p = head; p != nullptr; p = p->next) {
            // if (p->waste.load(memory_order_acquire)) continue;
            if (p->key != hashKey) continue;
            if (auto alive = p->value.lock()) {
                delete newEntry;
                return;
            } else if (p->value.expired()) {
                p->value = weak_ptr<QMDDNode>(node);
                delete newEntry;
                return;
            // } else {
            //     p->waste.store(true, memory_order_release);
            }
        }
        newEntry->next = head;
        // cout << "Inserting key: " << hashKey << " at index: " << idx << endl;
        if (cas((void**)&this->table_[idx], head, newEntry)) break;
        // cout << "CAS failed for key: " << hashKey << " at index: " << idx << ", retrying..." << endl;
        boost::this_fiber::yield();
        // loop_count++;
        // if (loop_count > 100) {
        //     std::cerr << "UniqueTable::insert: CAS loop too many times! idx=" << idx << std::endl;
        // //     abort();
        // }
    }
}

shared_ptr<QMDDNode> UniqueTable::find(int64_t hashKey) const {
    size_t idx = hash(hashKey);
    Entry* head = this->table_[idx].load(memory_order_acquire);
    for (Entry* p = head; p != nullptr; p = p->next) {
        if (p->key != hashKey) continue;
        if (auto alive = p->value.lock()) {
            return alive;
        }
    }
    return nullptr;
}

int64_t UniqueTable::hash(int64_t key) const {
    return key & (this->tableSize_ - 1);
}

void UniqueTable::throwAway(Entry* e) {
    Entry* old;
    do {
        old = this->dustelBoxHead_.load(memory_order_relaxed);
        e->next = old;
    } while (!this->dustelBoxHead_.compare_exchange_weak(
        old, e,
        memory_order_release,
        memory_order_relaxed));
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
            if (auto alive = p->value.lock()) {
                cout << "    " << *alive << endl;
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
            if (auto alive = p->value.lock()) {
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

// UniqueTable::UniqueTable() {
//     PARAMETER.load();

//     const int cfg = PARAMETER.table.size;

//     if (cfg > 0) {
//         const size_t expectedTotalEntries = static_cast<size_t>(cfg);
//         const size_t perShard = (expectedTotalEntries + kShardCount - 1) / kShardCount;
//         const size_t reservePerShard = static_cast<size_t>(perShard / kMaxLoadFactor) + 1;

//         for (auto& s : shards_) {
//             s.map.max_load_factor(kMaxLoadFactor);
//             s.map.reserve(reservePerShard);
//         }
//     } else {
//         for (auto& s : shards_) {
//             s.map.max_load_factor(kMaxLoadFactor);
//         }
//     }
// }

// UniqueTable& UniqueTable::getInstance() {
//     static UniqueTable instance;
//     return instance;
// }

// void UniqueTable::insert(int64_t hashKey, shared_ptr<QMDDNode> node) {
//     auto& sh = shards_[shardIndex(hashKey)];
//     std::unique_lock<std::shared_mutex> lock(sh.mtx, std::defer_lock);
//     while (!lock.try_lock()) {
//         boost::this_fiber::yield();
//     }

//     sh.map.try_emplace(hashKey, std::move(node));
// }

// shared_ptr<QMDDNode> UniqueTable::find(int64_t hashKey) const {
//     auto& sh = shards_[shardIndex(hashKey)];
//     std::shared_lock<std::shared_mutex> lock(sh.mtx, std::defer_lock);
//     while (!lock.try_lock()) {
//         boost::this_fiber::yield();
//     }

//     auto it = sh.map.find(hashKey);
//     return (it == sh.map.end()) ? nullptr : it->second;
// }

// int UniqueTable::getTotalEntryCount() const {
//     int total = 0;
//     for (auto& sh : shards_) {
//         std::lock_guard<std::shared_mutex> lock(sh.mtx);
//         total += static_cast<int>(sh.map.size());
//     }
//     return total;
// }

// void UniqueTable::printNodeNum() const {
//     int validEntries = 0;
//     int invalidEntries = 0;
//     for (auto& sh : shards_) {
//         std::lock_guard<std::shared_mutex> lock(sh.mtx);
//         for (auto& kv : sh.map) {
//             if (kv.second) validEntries++;
//             else invalidEntries++;
//         }
//     }
//     cout << "Total entries: " << (validEntries + invalidEntries) << endl;
//     cout << "Valid entries: " << validEntries << endl;
//     cout << "Invalid entries: " << invalidEntries << endl;
//     cout << "Shard count: " << kShardCount << endl;
// }

// void UniqueTable::printAllEntries() const {
//     for (size_t si = 0; si < kShardCount; ++si) {
//         auto& sh = shards_[si];
//         std::lock_guard<std::shared_mutex> lock(sh.mtx);
//         if (sh.map.empty()) continue;
//         cout << "Shard: " << si << endl;
//         for (auto& [k, v] : sh.map) {
//             cout << "  Key: " << k << endl;
//             if (v) cout << "    " << *v << endl;
//             else cout << "    Null node" << endl;
//         }
//     }
// }