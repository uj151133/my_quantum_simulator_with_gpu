#include "operationCache.hpp"

OperationCache& OperationCache::getInstance() {
    thread_local OperationCache instance;
    return instance;
}


void OperationCache::insert(size_t cacheKey, OperationResult result) {
    unique_lock<shared_mutex> lock(cacheMutex);
    cache[cacheKey] = result;
}


OperationResult OperationCache::find(size_t cacheKey) const {
    shared_lock<shared_mutex> lock(cacheMutex);
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