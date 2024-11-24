#ifndef OPERATIONCACHE_HPP
#define OPERATIONCACHE_HPP

#include <unordered_map>
#include <cstring>
#include <shared_mutex>
#include "qmdd.hpp"
#include "../common/calculation.hpp"
using namespace std;

class OperationCache {
private:
    unordered_map<size_t, OperationResult> cache;
    mutable shared_mutex cacheMutex;
    OperationCache() = default;

public:
    OperationCache(const OperationCache&) = delete;
    OperationCache& operator=(const OperationCache&) = delete;
    static OperationCache& getInstance();
    void insert(size_t cacheKey, OperationResult result);
    OperationResult find(size_t cacheKey) const;
    void printAllEntries() const;
};

#endif
