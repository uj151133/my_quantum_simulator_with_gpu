// #include "uniqueTable.hpp"

// UniqueTable::UniqueTable() : {
//     // #ifdef __APPLE__
//     //     CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
//     // #elif __linux__
//     //     CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
//     // #else
//     //     #error "Unsupported operating system"
//     // #endif
    
//     // tableSize = CONFIG.table.size;
//     table.reserve(tableSize);
// }

// size_t UniqueTable::hash(size_t hashKey) const {
//     return hashKey % tableSize;
// }

// UniqueTable& UniqueTable::getInstance() {
//     static UniqueTable instance;
//     return instance;
// }

// // void UniqueTable::insert(size_t hashKey, shared_ptr<QMDDNode> node) {
// //     auto waitStart = chrono::high_resolution_clock::now();
// //     while (!tableMutex.try_lock()) {
// //         boost::this_fiber::yield();
// //     }
// //     auto waitEnd = chrono::high_resolution_clock::now();
// //     totalWaitTime += chrono::duration_cast<chrono::microseconds>(waitEnd - waitStart);
// //     // totalWaitCount++;
// //     unique_lock<shared_mutex> lock(tableMutex, adopt_lock);
// //     size_t index = hash(hashKey);
// //     auto it = table.find(index);
// //     if (it != table.end()) {
// //         for (auto& existingEntry : it->second) {
// //             if (existingEntry.key == hashKey) {
// //                 if (existingEntry.value.lock() == node) return;
// //                 else if (existingEntry.value.expired()) {
// //                     existingEntry.value = weak_ptr<QMDDNode>(node);
// //                     return;
// //                 }
// //             }
// //         }
// //     }
// //     table[index].push_back(Entry(hashKey, weak_ptr<QMDDNode>(node)));
// //     return;
// // }

// void UniqueTable::insert(size_t hashKey, shared_ptr<QMDDNode> node) {
//     while (true) {
//         bool locked = tableMutex.try_lock();
        
//         if (locked) {
//             unique_lock<shared_mutex> lock(tableMutex, adopt_lock);
//             size_t index = hash(hashKey);
//             auto it = table.find(index);
//             if (it != table.end()) {
//                 for (auto& existingEntry : it->second) {
//                     if (existingEntry.key == hashKey and existingEntry.value == node) return
//                 }
//             }
//             table[index].push_back(Entry(hashKey, node));
//             return;
//         }
//         boost::this_fiber::yield();
//     }
// }

// // shared_ptr<QMDDNode> UniqueTable::find(size_t hashKey) const {
// //     auto waitStart = chrono::high_resolution_clock::now();
// //     while (!tableMutex.try_lock_shared()) {
// //         boost::this_fiber::yield();
// //     }
// //     auto waitEnd = chrono::high_resolution_clock::now();
// //     totalWaitTime += chrono::duration_cast<chrono::microseconds>(waitEnd - waitStart);
// //     // totalWaitCount++;
// //     shared_lock<shared_mutex> lock(tableMutex, adopt_lock);
// //     size_t index = hash(hashKey);
// //     auto it = table.find(index);
// //     if (it != table.end()) {
// //         for (const auto& entry : it->second) {
// //             if (entry.key == hashKey) {
// //                 return entry.value.lock();
// //             }
// //         }
// //     }
// //     return nullptr;
// // }

// shared_ptr<QMDDNode> UniqueTable::find(size_t hashKey) const {
//     while (true) {
//         bool locked = tableMutex.try_lock_shared();
        
//         if (locked) {
//             shared_lock<shared_mutex> lock(tableMutex, adopt_lock);
//             size_t index = hash(hashKey);
//             auto it = table.find(index);
//             if (it != table.end()) {
//                 for (const auto& entry : it->second) {
//                     if (entry.key == hashKey) {
//                         return entry.value;
//                     }
//                 }
//             }
//             return nullptr;
//         }
//         boost::this_fiber::yield();
//     }
// }

// void UniqueTable::printAllEntries() const {
//     int validEntries = 0;
//     int invalidEntries = 0;
//     for (const auto& item : table) {
//         size_t index = item.first;
//         const auto& entries = item.second;

//         cout << "Index: " << index << endl;
        
//         for (const auto& entry : entries) {
//             cout << "  Key: " << entry.key << endl;
//             cout << "  Nodes: " << endl;
//             if (entry.value) {
//                 // const QMDDNode& node = *entry.value.lock();
//                 const QMDDNode& node = *entry.value;
//                 cout << "    " << node << endl;
//                 validEntries++;
//             } else {
//                 cout << "    Null node" << endl;
//                 invalidEntries++;
//             }
//         }
//         cout << endl;
//     }
//     cout << "Total entries: " << table.size() << endl;
//     cout << "Table size: " << tableSize << endl;
//     cout << "Valid entries: " << validEntries << endl;
//     cout << "Invalid entries: " << invalidEntries << endl;
//     cout << "Table bucket count: " << table.bucket_count() << endl;
// }

#include "uniqueTable.hpp"

extern "C" {
    #include "../atomic/atomic.h"
    }

UniqueTable::UniqueTable() : table(tableSize) {
    for (auto& entry : table) entry.store(nullptr, std::memory_order_relaxed);
}

UniqueTable& UniqueTable::getInstance() {
    static UniqueTable instance;
    return instance;
}

void UniqueTable::insert(size_t hashKey, std::shared_ptr<QMDDNode> node) {
    size_t idx = hash(hashKey);
    Entry* newEntry = new Entry(hashKey, node, nullptr);
    Entry* oldHead;
    while (true) {
        oldHead = table[idx].load(std::memory_order_acquire);
        for (Entry* p = oldHead; p != nullptr; p = p->next) {
            if (p->key == hashKey && p->value == node) {
                delete newEntry;
                return;
            }
        }
        newEntry->next = oldHead;
        if (cas_arm64((void**)&table[idx], oldHead, newEntry)) break;
        // if(table[idx].compare_exchange_strong(oldHead, newEntry, std::memory_order_release, std::memory_order_relaxed)) break;
        boost::this_fiber::yield();
    }
}

shared_ptr<QMDDNode> UniqueTable::find(size_t hashKey) const {
    size_t idx = hash(hashKey);
    Entry* head = table[idx].load(std::memory_order_acquire);
    for (Entry* p = head; p != nullptr; p = p->next) {
        if (p->key == hashKey) {
            return p->value;
        }
    }
    return nullptr;
}

// 実装ファイル
size_t UniqueTable::hash(size_t key) const {
    return key & (tableSize - 1);
}

void UniqueTable::printAllEntries() const {
    int validEntries = 0;
    int invalidEntries = 0;
    for (size_t idx = 0; idx < tableSize; ++idx) {
        Entry* head = table[idx].load(std::memory_order_acquire);
        if (!head) continue;
        std::cout << "Index: " << idx << std::endl;
        for (Entry* p = head; p != nullptr; p = p->next) {
            std::cout << "  Key: " << p->key << std::endl;
            std::cout << "  Nodes: " << std::endl;
            if (p->value) {
                std::cout << "    " << *p->value << std::endl;
                validEntries++;
            } else {
                std::cout << "    Null node" << std::endl;
                invalidEntries++;
            }
        }
        std::cout << std::endl;
    }
    std::cout << "Total entries: (unknown in vector mode)" << std::endl;
    std::cout << "Table size: " << tableSize << std::endl;
    std::cout << "Valid entries: " << validEntries << std::endl;
    std::cout << "Invalid entries: " << invalidEntries << std::endl;
    std::cout << "Table bucket count: " << tableSize << std::endl;
}