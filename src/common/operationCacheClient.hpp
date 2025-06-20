#ifndef OPERATIONCACHECLIENT_HPP
#define OPERATIONCACHECLIENT_HPP

#include <graal_isolate.h>
#include <stdexcept>
#include <memory>
#include <complex>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <cstdlib>
#include <optional>
#include "../models/qmdd.hpp"

using namespace std;

extern "C" {
    void cacheInsert(graal_isolatethread_t* thread, long long key, 
                     double real, double imag, long long uniqueTableKey);
    void* cacheFind(graal_isolatethread_t* thread, long long key);
}

class OperationCacheClient {
private:
    graal_isolate_t* isolate;
    
    // スレッドローカルなIsolateThreadを管理
    thread_local static graal_isolatethread_t* thread_local_thread;
    static mutex isolate_mutex;
    
    graal_isolatethread_t* getThreadLocalThread();

public:
    OperationCacheClient();
    ~OperationCacheClient();
    
    // void insert(long long key, const complex<double>& weight, long long uniqueTableKey);
    // bool find(long long key, complex<double>& weight, long long& uniqueTableKey);
    void insert(long long key, const QMDDEdge& edge);
    optional<QMDDEdge> find(long long key);
    static OperationCacheClient& getInstance();
};

#endif