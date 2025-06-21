#ifndef OPERATIONCACHE_HPP
#define OPERATIONCACHE_HPP

#include <unordered_map>
#include <cstring>
#include <shared_mutex>
#include <mutex>
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
    unordered_map<long long, OperationResult> cache;
    size_t cacheSize;
    static mutex instancesMutex;
    static vector<OperationCache*> instances;
    // mutable shared_mutex cacheMutex;
    OperationCache();
    ~OperationCache();

public:
    OperationCache(const OperationCache&) = delete;
    OperationCache& operator=(const OperationCache&) = delete;
    static OperationCache& getInstance();
    void insert(long long cacheKey, OperationResult result);
    void clear();
    static void clearAllCaches();
    OperationResult find(long long cacheKey) const;
    void printAllEntries() const;
};

#endif