#ifndef UNIQUETABLE_HPP
#define UNIQUETABLE_HPP

#include <atomic>
#include <array>
#include <vector>
#include <iostream>
#include "../common/parameter.hpp"
#include "qmdd.hpp"

using namespace std;

class UniqueTable {
private:
    static inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
        __asm__ __volatile__("yield");
#endif
    }
    struct SpinLock {
        std::atomic_flag f = ATOMIC_FLAG_INIT;
        void lock()   { while (this->f.test_and_set(std::memory_order_acquire)) { cpuRelax(); } }
        void unlock() { this->f.clear(std::memory_order_release); }
    };
    static constexpr size_t LOCK_SHARDS = 8192;
    const int64_t tableSize_;
    vector<atomic<QMDDNode*>> table_;
    mutable array<SpinLock, LOCK_SHARDS> locks_;
    UniqueTable();
    int64_t hash(int64_t hashKey) const;
    size_t shard(int64_t idx) const { return static_cast<size_t>(idx) & (LOCK_SHARDS - 1); }

public:
    UniqueTable(const UniqueTable&) = delete;
    UniqueTable& operator=(const UniqueTable&) = delete;
    static UniqueTable& getInstance();

    QMDDNode* insert(int64_t hashKey, QMDDNode* node);
    QMDDNode* find(int64_t hashKey);

    void release(QMDDNode* node);

    void printAllEntries() const;
    void printNodeNum() const;
    int getTotalEntryCount() const;
};

#endif
