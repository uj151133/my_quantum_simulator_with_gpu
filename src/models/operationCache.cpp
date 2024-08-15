#include "operationCache.hpp"

OperationCache& OperationCache::getInstance() {
    static OperationCache instance;
    return instance;
}

void OperationCache::insert(size_t cacheKey, OperationResult result) {
    cache[cacheKey] = result;
}

OperationResult OperationCache::find(size_t cacheKey) const {
    auto it = cache.find(cacheKey);
    if (it != cache.end()) {
        return it->second;
    }
    return OperationResult();
}

void OperationCache::printAllEntries() const {
    for (const auto& entry : cache) {
        size_t key = entry.first;
        const OperationResult& result = entry.second;

        cout << "Key: " << key << endl;
        cout << "Result: " << result.first << " " << result.second << endl;
    }
}