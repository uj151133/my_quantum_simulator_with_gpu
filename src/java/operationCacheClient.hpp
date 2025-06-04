#ifndef OPERATION_CACHE_CLIENT_HPP
#define OPERATION_CACHE_CLIENT_HPP

#include <optional>
#include <string>
#include <cstdint>
#include "../models/qmdd.hpp"
#include "./graal_isolate.h"

using namespace std;

class OperationCacheClient {
private:
    void* library_handle;

    // typedef int (*GraalCreateIsolateFunc)(void**, void**);
    typedef int (*GraalCreateIsolateFunc)(graal_isolate_t**, graal_isolatethread_t**);
    typedef int (*GraalTearDownIsolateFunc)(graal_isolatethread_t*);
    
    // 関数ポインタ型定義
    typedef int (*InsertFunc)(long long, double, double, long long);
    typedef int (*FindFunc)(long long, char*);
    typedef int (*ContainsFunc)(long long);
    typedef int64_t (*SizeFunc)();
    typedef int (*ClearFunc)();
    

    GraalCreateIsolateFunc graal_create_isolate;
    GraalTearDownIsolateFunc graal_tear_down_isolate;
    // 関数ポインタ
    InsertFunc insert_func;
    FindFunc find_func;
    ContainsFunc contains_func;
    SizeFunc size_func;
    ClearFunc clear_func;
    
    static constexpr size_t BUFFER_SIZE = 1024;

    graal_isolatethread_t* isolate_thread;
    graal_isolate_t* isolate;
    bool graal_initialized;
    
    // プライベートコンストラクタ
    OperationCacheClient(const string& library_path);
    
    void loadLibrary(const string& library_path);

public:
    // コピー・ムーブ禁止
    OperationCacheClient(const OperationCacheClient&) = delete;
    OperationCacheClient& operator=(const OperationCacheClient&) = delete;
    OperationCacheClient(OperationCacheClient&&) = delete;
    OperationCacheClient& operator=(OperationCacheClient&&) = delete;
    
    ~OperationCacheClient();
    
    // シングルトンインスタンス取得
    static OperationCacheClient& getInstance();
    static void initialize(const string& library_path = "/Users/mitsuishikaito/my_quantum_simulator_with_gpu/src/java/liboperation-cache.dylib");

    // キャッシュ操作メソッド
    void insert(long long key, OperationResult result);
    OperationResult find(long long key);
    bool contains(long long key);
    int64_t size();
    bool clear();
    
    // デバッグ情報
    void printStatus();
};

#endif