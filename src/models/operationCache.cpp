#include "operationCache.hpp"

OperationCache& OperationCache::getInstance() {
    static OperationCache instance;
    return instance;
}

void OperationCache::insert(size_t cacheKey, OperationResult result) {
    // cache[cacheKey] = result;
    lock_guard<mutex> lock(cacheMutex);
    auto it = cache.find(cacheKey);
    if (it != cache.end()) {
        auto currentValue = it->second;
        if (__sync_bool_compare_and_swap(&it->second, currentValue, result)) {
            cout << "Successfully updated cache for key: " << cacheKey << endl;
        } else {
            cerr << "Failed to update cache for key: " << cacheKey << ". Retrying..." << endl;
                cache.insert({cacheKey, result});
        }
    } else {
        cache.insert({cacheKey, result});
        cout << "Inserted new cache entry for key: " << cacheKey << std::endl;
    }
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