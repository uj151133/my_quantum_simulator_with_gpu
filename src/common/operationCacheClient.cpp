#include "operationCacheClient.hpp"
#include <boost/fiber/fss.hpp>
#include <atomic>

// static auto alienacheInsert = cacheInsert;
// static auto alienCacheFind = cacheFind;

thread_local graal_isolatethread_t* OperationCacheClient::thread_local_thread = nullptr;
mutex OperationCacheClient::isolateMutex_;

struct alignas(8) NativeOperationResult24 {
    double real;
    double imag;
    int64_t uniqueTableKey;
};

OperationCacheClient::OperationCacheClient() : flsCapacity_(PARAMETER.cache.FLSSize), isolate_(nullptr) {
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
    return this->initializeNewThread();
}

graal_isolatethread_t* OperationCacheClient::initializeNewThread() {
    if (this->isolate_ == nullptr) {
        throw runtime_error("GraalVM isolate is not initialized");
    }

    if (graal_attach_thread(this->isolate_, &thread_local_thread) != 0) {
        throw runtime_error("Failed to attach thread to GraalVM isolate");
    }
    
    return thread_local_thread;
}

OperationCacheClient::FlsEntry& OperationCacheClient::fls() {
    static boost::fibers::fiber_specific_ptr<FlsEntry> slot;
    FlsEntry* p = slot.get();
    if (!p) { p = new FlsEntry(); slot.reset(p); }
    return *p;
}

optional<QMDDEdge> OperationCacheClient::find(int64_t key, bool useLocal) {
    if (useLocal) {
        auto& d = fls();
        auto it = d.cache.find(key);
        if (it == d.cache.end()) return nullopt;
        return it->second;
    }
    return this->findGlobal(key);
}

void OperationCacheClient::insert(int64_t key, const QMDDEdge& edge, bool useLocal) {
    if (useLocal) {
        const int64_t cap = this->flsCapacity_;
        auto& d = fls();
        auto it = d.cache.find(key);
        if (it != d.cache.end()) { it->second = edge; return; }
        d.cache.emplace(key, edge);
        if (cap == 0 || (int64_t)d.cache.size() > cap) d.cache.clear();
        return;
    }
    this->insertGlobal(key, edge);
}

void OperationCacheClient::clearFiberLocalCache() {
    auto& d = fls();
    d.cache.clear();
}

__attribute__((hot, flatten))
void OperationCacheClient::insertGlobal(int64_t key, const QMDDEdge& edge) {
    graal_isolatethread_t* thread = this->getThreadLocalThread();
    cacheInsert(thread, key, edge.weight.real(), edge.weight.imag(), edge.key_);
}

__attribute__((hot, flatten))
optional<QMDDEdge> OperationCacheClient::findGlobal(int64_t key) {
    graal_isolatethread_t* thread = this->getThreadLocalThread();
    NativeOperationResult24 out{};
    const int found = cacheFind(thread, key, static_cast<void*>(&out));

    if (!found || UniqueTable::getInstance().find(out.uniqueTableKey) == nullptr) {
        return nullopt;
    }

    return make_optional<QMDDEdge>(
        complex<double>(out.real, out.imag),
        out.uniqueTableKey,
        SonKind::QMDDNode
    );
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

void OperationCacheClient::flushFiberLocalToGlobal() {
    auto& d = fls();
    if (d.cache.empty()) return;
    for (const auto& kv : d.cache) {
        this->insertGlobal(kv.first, kv.second);
    }
    d.cache.clear();
}

void OperationCacheClient::saveCacheToSQLite() {
    graal_isolatethread_t* thread = this->getThreadLocalThread();
    if (thread == nullptr) {
        thread = this->initializeNewThread();
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