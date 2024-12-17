#ifndef OPERATIONCACHE_HPP
#define OPERATIONCACHE_HPP

#include <unordered_map>
#include <cstring>
// #include <shared_mutex>

#include "../common/config.hpp"
#include "qmdd.hpp"
#include "../common/calculation.hpp"
using namespace std;

struct CacheEntry {
    OperationResult result;
    std::chrono::steady_clock::time_point lastAccess;
};

class OperationCache {
private:
    unordered_map<size_t, OperationResult> cache;
    // std::list<size_t> lruList;
    size_t cacheSize;
    // mutable shared_mutex cacheMutex;c
    OperationCache();
    // void updateLRU(size_t cacheKey);

public:
    OperationCache(const OperationCache&) = delete;
    OperationCache& operator=(const OperationCache&) = delete;
    static OperationCache& getInstance();
    void insert(size_t cacheKey, OperationResult result);
    void clear();
    OperationResult find(size_t cacheKey) const;
    void printAllEntries() const;
};

#endif
