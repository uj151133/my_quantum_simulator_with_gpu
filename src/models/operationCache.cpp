#include "operationCache.hpp"

OperationCache::OperationCache() {
    // #ifdef __APPLE__
    //     CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    // #elif __linux__
    //     CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    // #else
    //     #error "Unsupported operating system"
    // #endif

    // cacheSize = 1000000 / 4;
    // cache.reserve(cacheSize);
}

// void OperationCache::updateLRU(size_t cacheKey) {
//     auto it = std::find(lruList.begin(), lruList.end(), cacheKey);
//     if (it != lruList.end()) {
//         lruList.erase(it);
//     }
//     lruList.push_front(cacheKey);
// }

OperationCache& OperationCache::getInstance() {
    thread_local OperationCache instance;
    return instance;
}

void OperationCache::insert(size_t cacheKey, OperationResult result) {
    // unique_lock<shared_mutex> lock(cacheMutex);
    if (cache.size() >= 250000) {
        // size_t oldestKey = lruList.back();
        // lruList.pop_back();
        // cache.erase(oldestKey);
        return;
    }
    // cache[cacheKey] = CacheEntry{result, chrono::steady_clock::now()};
    cache[cacheKey] = result;
    return;
    // updateLRU(cacheKey);
}

void OperationCache::clear() {
    // unique_lock<shared_mutex> lock(cacheMutex);
    cache.clear();
}


OperationResult OperationCache::find(size_t cacheKey) const {
    // shared_lock<shared_mutex> lock(cacheMutex);
    auto it = cache.find(cacheKey);
    if (it != cache.end()) {
        return it->second;
    }
    return OperationResult();
}

void OperationCache::printAllEntries() const {
    for (const auto& item : cache) {
        size_t key = item.first;
        const OperationResult entry = item.second;

        cout << "Key: " << key << endl;
        cout << "Result: " << entry.first << " " << entry.second << endl;

        // cout << "Last access: " << entry.lastAccess.time_since_epoch().count() << endl;
    }

    cout << "Number of valid entries: " << cache.size() << endl;
    // cout << "Cache size: " << cacheSize << endl;
}