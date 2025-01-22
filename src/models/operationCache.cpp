#include "operationCache.hpp"

mutex OperationCache::instancesMutex;
vector<OperationCache*> OperationCache::instances;


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
    lock_guard<mutex> lock(instancesMutex);
    instances.push_back(this);
}

OperationCache::~OperationCache() {
    lock_guard<mutex> lock(instancesMutex);
    auto it = std::find(instances.begin(), instances.end(), this);
    if (it != instances.end()) {
        instances.erase(it);
    }
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
    // while (!cacheMutex.try_lock()) {
    //     boost::this_fiber::yield();
    // }
    // unique_lock<shared_mutex> lock(cacheMutex, adopt_lock);
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

void OperationCache::clearAllCaches() {
    lock_guard<mutex> lock(instancesMutex);
    for (auto* cache : instances) {
        cache->clear();
    }
}

OperationResult OperationCache::find(size_t cacheKey) const {
    // while (!cacheMutex.try_lock_shared()) {
    //     boost::this_fiber::yield();
    // }
    // shared_lock<shared_mutex> lock(cacheMutex, adopt_lock);
    auto it = cache.find(cacheKey);
    if (it != cache.end()) {
        return it->second;
    }
    return OperationResult();
}

void OperationCache::printAllEntries() const {
    cout << "Cache Entries:" << endl;

    for (const auto& item : cache) {
        size_t key = item.first;
        const OperationResult entry = item.second;

        cout << "Key: " << key << endl;
        cout << "Result: " << entry.first << " " << entry.second << endl;

        // cout << "Last access: " << entry.lastAccess.time_since_epoch().count() << endl;
    }

    // cout << "Cache size: " << cacheSize << endl;
}