#include "operationCacheClient.hpp"
#include <iostream>
#include <cstdlib>

// スレッドローカル変数の定義
thread_local graal_isolatethread_t* OperationCacheClient::thread_local_thread = nullptr;
mutex OperationCacheClient::isolate_mutex;

OperationCacheClient::OperationCacheClient() : isolate(nullptr) {
    lock_guard<mutex> lock(isolate_mutex);
    if (graal_create_isolate(nullptr, &isolate, nullptr) != 0) {
        throw runtime_error("Failed to create GraalVM isolate for OperationCache");
    }
}

OperationCacheClient::~OperationCacheClient() {
    if (thread_local_thread != nullptr) {
        graal_detach_thread(thread_local_thread);
        thread_local_thread = nullptr;
    }

    if (isolate) {
        graal_isolatethread_t* main_thread = nullptr;
        if (graal_attach_thread(isolate, &main_thread) == 0) {
            graal_tear_down_isolate(main_thread);
        }
        isolate = nullptr;
    }
}

graal_isolatethread_t* OperationCacheClient::getThreadLocalThread() {
    if (thread_local_thread == nullptr) {
        if (graal_attach_thread(isolate, &thread_local_thread) != 0) {
            throw runtime_error("Failed to attach thread to GraalVM isolate");
        }
    }
    return thread_local_thread;
}

void OperationCacheClient::insert(long long key, const QMDDEdge& edge) {
    graal_isolatethread_t* thread = this->getThreadLocalThread();
    cacheInsert(thread, key, edge.weight.real(), edge.weight.imag(), edge.uniqueTableKey);
}

optional<QMDDEdge> OperationCacheClient::find(long long key) {
    graal_isolatethread_t* thread = this->getThreadLocalThread();
    void* ptr = cacheFind(thread, key);

    if (ptr == nullptr) {
        return nullopt;
    }

    double* data = static_cast<double*>(ptr);

    auto result = make_optional<QMDDEdge>(
        complex<double>(data[0], data[1]),
        *reinterpret_cast<long long*>(&data[2])
    );

    free(ptr);
    return result;
}

void OperationCacheClient::cleanup() {
    lock_guard<mutex> lock(isolate_mutex);
    
    if (thread_local_thread != nullptr) {
        graal_detach_thread(thread_local_thread);
        thread_local_thread = nullptr;
    }
    
    if (isolate) {
        graal_isolatethread_t* main_thread = nullptr;
        if (graal_attach_thread(isolate, &main_thread) == 0) {
            graal_tear_down_isolate(main_thread);
        }
        isolate = nullptr;
    }
}

OperationCacheClient& OperationCacheClient::getInstance() {
    static OperationCacheClient instance;
    return instance;
}