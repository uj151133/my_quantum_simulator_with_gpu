#include "operationCacheClient.hpp"

// static auto alienacheInsert = cacheInsert;
// static auto alienCacheFind = cacheFind;

thread_local graal_isolatethread_t* OperationCacheClient::thread_local_thread = nullptr;
// thread_local unordered_map<int64_t, QMDDEdge> OperationCacheClient::TLSCache_;
thread_local unordered_map<int64_t, OperationCacheClient::TLSEntry> OperationCacheClient::TLSCache_;
thread_local std::list<int64_t> OperationCacheClient::TLSLru_;

mutex OperationCacheClient::isolateMutex_;

struct alignas(8) NativeOperationResult24 {
    double magnitude;
    double angle;
    int64_t uniqueTableKey;
};

OperationCacheClient::OperationCacheClient() : tlsCapacity_(PARAMETER.cache.TLSSize), isolate_(nullptr) {
    lock_guard<mutex> lock(this->isolateMutex_);
    if (graal_create_isolate(nullptr, &this->isolate_, nullptr) != 0) {
        throw runtime_error("Failed to create GraalVM isolate for OperationCache");
    }
}

OperationCacheClient::~OperationCacheClient() {
    if (thread_local_thread != nullptr) {
        graal_detach_thread(thread_local_thread);
        thread_local_thread = nullptr;
    }

    if (this->isolate_) {
        graal_isolatethread_t* main_thread = nullptr;
        if (graal_attach_thread(this->isolate_, &main_thread) == 0) {
            graal_tear_down_isolate(main_thread);
        }
        this->isolate_ = nullptr;
    }
}

__attribute__((hot, flatten))
inline graal_isolatethread_t* OperationCacheClient::getThreadLocalThread() {
    if (thread_local_thread != nullptr) {
            return thread_local_thread;
        }
    return initializeNewThread();
}

graal_isolatethread_t* OperationCacheClient::initializeNewThread() {
    if (this->isolate_ == nullptr) {
        throw runtime_error("GraalVM isolate is not initialized");
    }

    if (graal_attach_thread(this->isolate_, &thread_local_thread) != 0) {
        throw runtime_error("Failed to attach thread to GraalVM isolate");
    }
    
    #ifdef DEBUG
    cout << "New thread attached to GraalVM isolate: " << thread_local_thread << endl;
    #endif
    
    return thread_local_thread;
}

optional<QMDDEdge> OperationCacheClient::find(int64_t key, bool useTLS) {
    if (useTLS) {
        auto it = TLSCache_.find(key);
        if (it == TLSCache_.end()) return nullopt;
        tlsTouch_(it);
        return it->second.edge;
    }
    return this->findGlobal(key);
}

void OperationCacheClient::insert(int64_t key, const QMDDEdge& edge, bool useTLS) {
    if (useTLS) {
        const int64_t cap = this->tlsCapacity_;

        auto it = TLSCache_.find(key);
        if (it != TLSCache_.end()) {
            it->second.edge = edge;
            tlsTouch_(it);
            return;
        }

        TLSLru_.push_front(key);
        TLSCache_.emplace(key, TLSEntry{edge, TLSLru_.begin()});
        tlsEvict_();
        return;
    }
    this->insertGlobal(key, edge);
}

__attribute__((hot, flatten))
void OperationCacheClient::insertGlobal(int64_t key, const QMDDEdge& edge) {
    graal_isolatethread_t* thread = this->getThreadLocalThread();
    cacheInsert(thread, key, edge.magnitude, edge.angle, edge.key_);
}

__attribute__((hot, flatten))
optional<QMDDEdge> OperationCacheClient::findGlobal(int64_t key) {
    graal_isolatethread_t* thread = this->getThreadLocalThread();
    NativeOperationResult24 out{};
    const int found = cacheFind(thread, key, static_cast<void*>(&out));

    if (!found || UniqueTable::getInstance().find(out.uniqueTableKey) == nullptr) {
        return nullopt;
    }

    return QMDDEdge({out.magnitude, out.angle}, out.uniqueTableKey, SonKind::QMDDNode);
}

inline void OperationCacheClient::tlsTouch_(unordered_map<int64_t, TLSEntry>::iterator it) {
    // move to front (MRU), O(1)
    TLSLru_.splice(TLSLru_.begin(), TLSLru_, it->second.it);
    it->second.it = TLSLru_.begin();
}

inline void OperationCacheClient::tlsEvict_() {
    const int64_t cap = this->tlsCapacity_;
    if (cap == 0) {
        TLSCache_.clear();
        TLSLru_.clear();
        return;
    }
    while (TLSCache_.size() > cap) {
        const int64_t victim = TLSLru_.back(); // LRU
        TLSLru_.pop_back();
        TLSCache_.erase(victim);
    }
}

void OperationCacheClient::cleanup() {
    lock_guard<mutex> lock(this->isolateMutex_);
    
    if (thread_local_thread != nullptr) {
        graal_detach_thread(thread_local_thread);
        thread_local_thread = nullptr;
    }
    
    if (this->isolate_) {
        graal_isolatethread_t* main_thread = nullptr;
        if (graal_attach_thread(this->isolate_, &main_thread) == 0) {
            graal_tear_down_isolate(main_thread);
        }
        this->isolate_ = nullptr;
    }
}

OperationCacheClient& OperationCacheClient::getInstance() {
    static OperationCacheClient* instance = new OperationCacheClient();
    return *instance;
}

void OperationCacheClient::flushThreadLocalToGlobal() {
    if (this->TLSCache_.empty()) return;
    for (const auto& kv : this->TLSCache_) {
        insertGlobal(kv.first, kv.second.edge);
    }
    this->TLSCache_.clear();
    TLSLru_.clear();
}

void OperationCacheClient::saveCacheToSQLite() {
    graal_isolatethread_t* thread = getThreadLocalThread();
    if (thread == nullptr) {
        thread = initializeNewThread();
        if (thread == nullptr) {
            throw runtime_error("Failed to create isolate thread for SQLite save");
        }
    }

    lock_guard<mutex> lock(this->isolateMutex_);
    
    try {
        ::saveCacheToSQLite(thread);
        cout << "OperationCacheClient: Cache saved to SQLite successfully" << endl;
    } catch (const exception& e) {
        throw runtime_error("Failed to save cache to SQLite: " + string(e.what()));
    }
}