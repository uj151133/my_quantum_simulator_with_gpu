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
#include "parameter.hpp"
#include "../models/qmdd.hpp"
#include "../models/uniqueTable.hpp"

using namespace std;



extern "C" {
    void cacheInsert(graal_isolatethread_t* thread, int64_t key, double real, double imag, int64_t uniqueTableKey);
    int cacheFind(graal_isolatethread_t* thread, int64_t key, void* out);
    void saveCacheToSQLite(graal_isolatethread_t* thread);
}

class OperationCacheClient {
private:
    graal_isolate_t* isolate_;
    thread_local static graal_isolatethread_t* thread_local_thread;
    static mutex isolateMutex_;
    inline graal_isolatethread_t* getThreadLocalThread();
    graal_isolatethread_t* initializeNewThread();

    // 演算キャッシュの 1 件。ノードは生かさない（参照カウントを持たない）。
    // 使うときにユニークテーブルでまだ生きているかを確かめる（DDSIM の演算キャッシュと同じ考え方）。
    // キャッシュが結果を抱え込むと、その子の密葉（GPU バッファ）まで解放されなくなるため。
    struct WeakEdge {
        complex<double> weight;
        int64_t key;          // ユニークテーブルのキー（0 = 持っていない）
    };
    struct FlsEntry {
        unordered_map<int64_t, WeakEdge> cache;
    };
    static FlsEntry& fls();
    const int64_t flsCapacity_;

    optional<QMDDEdge> findGlobal(int64_t key);
    void insertGlobal(int64_t key, complex<double> weight, int64_t uniqueTableKey);

public:
    OperationCacheClient();
    ~OperationCacheClient();

    optional<QMDDEdge> find(int64_t key, bool useLocal);
    void insert(int64_t key, const QMDDEdge& edge, bool useLocal);
    void clearFiberLocalCache();

    static OperationCacheClient& getInstance();
    void cleanup();
    void flushFiberLocalToGlobal();

    void saveCacheToSQLite();
};

#endif