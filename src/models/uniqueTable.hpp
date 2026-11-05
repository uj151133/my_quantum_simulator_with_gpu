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

    // 参照カウントが 0 でなければ +1 する（不可分）。0 なら何もせず false。
    // 「読んでから +1」に分けると、その間に最後の参照が外れて 0 になったノードを拾い直してしまい、
    // release() が 2 回走って二重に delete される。一度 0 になったノードは誰にも拾わせない。
    static bool tryRetain(QMDDNode* p) {
        uint32_t c = p->refCount.load(std::memory_order_acquire);
        while (c != 0) {
            if (p->refCount.compare_exchange_weak(c, c + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return true;
        }
        return false;
    }

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
