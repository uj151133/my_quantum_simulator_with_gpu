#ifndef UNIQUETABLE_HPP
#define UNIQUETABLE_HPP

#include <atomic>
#include <vector>
#include <memory>
#include <iostream>
// #include <shared_mutex>
#include "../common/parameter.hpp"
#include "qmdd.hpp"

using namespace std;

struct Entry {
    int64_t key;
    // shared_ptr<QMDDNode> value;
    weak_ptr<QMDDNode> value;
    Entry* next;
    // atomic<bool> waste;
    Entry(int64_t k, weak_ptr<QMDDNode> v, Entry* n=nullptr) : key(k), value(v), next(n) {}
};

class UniqueTable {
private:
    int64_t tableSize_;
    vector<atomic<Entry*>> table_;
    atomic<Entry*> dustelBoxHead_{nullptr};
    UniqueTable();
    int64_t hash(int64_t hashKey) const;
    void throwAway(Entry* e);

public:
    UniqueTable(const UniqueTable&) = delete;
    UniqueTable& operator=(const UniqueTable&) = delete;
    static UniqueTable& getInstance();
    void insert(int64_t hashKey, shared_ptr<QMDDNode> node);
    shared_ptr<QMDDNode> find(int64_t hashKey) const;
    void printAllEntries() const;
    void printNodeNum() const;
    int getTotalEntryCount() const;
};

#endif

// #ifndef UNIQUETABLE_HPP
// #define UNIQUETABLE_HPP

// #include <array>
// #include <cstdint>
// #include <memory>
// #include <mutex>
// #include <shared_mutex>
// #include <folly/container/F14Map.h>
// #include "../common/parameter.hpp"
// #include "qmdd.hpp"

// using namespace std;

// class UniqueTable {
// private:
//     static constexpr size_t kShardCount = 64; // 2の冪
//     static constexpr float kMaxLoadFactor = 0.80f;
//     using MapT = folly::F14FastMap<int64_t, shared_ptr<QMDDNode>>;

//     struct Shard {
//         mutable std::shared_mutex mtx;
//         MapT map;
//     };

//     std::array<Shard, kShardCount> shards_;

//     UniqueTable();

//     static inline uint64_t mix64(uint64_t x) {
//         // splitmix64
//         x += 0x9e3779b97f4a7c15ULL;
//         x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
//         x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
//         return x ^ (x >> 31);
//     }

//     inline size_t shardIndex(int64_t key) const {
//         return static_cast<size_t>(mix64(static_cast<uint64_t>(key)) & (kShardCount - 1));
//     }

// public:
//     UniqueTable(const UniqueTable&) = delete;
//     UniqueTable& operator=(const UniqueTable&) = delete;

//     static UniqueTable& getInstance();

//     void insert(int64_t hashKey, shared_ptr<QMDDNode> node);
//     shared_ptr<QMDDNode> find(int64_t hashKey) const;

//     void printAllEntries() const;
//     void printNodeNum() const;
//     int getTotalEntryCount() const;
// };


// #endif
