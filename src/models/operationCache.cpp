#include "operationCache.hpp"

OperationCache& OperationCache::getInstance() {
    static OperationCache instance;
    return instance;
}


void OperationCache::insert(size_t cacheKey, OperationResult result) {
    lock_guard<mutex> lock(cacheMutex);  // キャッシュ全体の操作を保護
    // auto it = cache.find(cacheKey);
    // if (it != cache.end()) {
        // compare_and_swap(it->second, it->second, result);  // CAS を適用
    // } else {
        cache[cacheKey] = result;  // 新規挿入
    // }
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