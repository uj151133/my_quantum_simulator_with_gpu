#ifndef OPERATIONCACHECLIENT_HPP
#define OPERATIONCACHECLIENT_HPP

#include <graal_isolate.h>
#include <stdexcept>
#include <memory>
#include <complex>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <cstdlib>
#include <memory_resource>
#include "../models/qmdd.hpp"

using namespace std;



extern "C" {
    void cacheInsert(graal_isolatethread_t* thread, int64_t key, double real, double imag, int64_t uniqueTableKey);
    void* cacheFind(graal_isolatethread_t* thread, int64_t key);
    void saveCacheToSQLite(graal_isolatethread_t* thread);
}

class OperationCacheClient {
private:
    graal_isolate_t* isolate;
    thread_local static graal_isolatethread_t* thread_local_thread;
    static mutex isolate_mutex;
    inline graal_isolatethread_t* getThreadLocalThread();
    graal_isolatethread_t* initializeNewThread();

    static thread_local unordered_map<int64_t, QMDDEdge> TLSCache;

    optional<QMDDEdge> findGlobal(int64_t key);
    void insertGlobal(int64_t key, const QMDDEdge& edge);

public:
    OperationCacheClient();
    ~OperationCacheClient();

    optional<QMDDEdge> find(int64_t key, bool useTLS);
    void insert(int64_t key, const QMDDEdge& edge, bool useTLS);

    static OperationCacheClient& getInstance();
    void cleanup();
    void flushThreadLocalToGlobal();
    void saveCacheToSQLite();
};

#endif