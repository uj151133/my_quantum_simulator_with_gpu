#ifndef UNIQUETABLE_HPP
#define UNIQUETABLE_HPP

#include <unordered_map>
#include <vector>
#include <memory>
#include "qmdd.hpp"

using namespace std;

class OperationCache {
private:
    unordered_map<size_t, OperationResult> cache;
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
