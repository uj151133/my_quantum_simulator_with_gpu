#include "operationCacheClient.hpp"
#include <iostream>
#include <dlfcn.h>
#include <stdexcept>
#include <cstdio>

static const string DEFAULT_LIBRARY_PATH = "/Users/mitsuishikaito/my_quantum_simulator_with_gpu/src/java/liboperation-cache.dylib";
static string g_library_path = DEFAULT_LIBRARY_PATH;
static bool g_initialized = false;

// プライベートコンストラクタ
OperationCacheClient::OperationCacheClient(const string& library_path) 
    : library_handle(nullptr), graal_create_isolate(nullptr), graal_tear_down_isolate(nullptr),
      insert_func(nullptr), find_func(nullptr), contains_func(nullptr), 
      size_func(nullptr), clear_func(nullptr), isolate_thread(nullptr), 
      isolate(nullptr), graal_initialized(false) {
    loadLibrary(library_path);
}

void OperationCacheClient::loadLibrary(const string& library_path) {
    // 共有ライブラリをロード
    library_handle = dlopen(library_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!library_handle) {
        throw runtime_error(string("Cannot load library: ") + dlerror());
    }
    
    graal_create_isolate = (GraalCreateIsolateFunc) dlsym(library_handle, "graal_create_isolate");
    graal_tear_down_isolate = (GraalTearDownIsolateFunc) dlsym(library_handle, "graal_tear_down_isolate");
    
    // 既存の関数ポインタを取得
    insert_func = (InsertFunc) dlsym(library_handle, "operation_cache_insert");
    find_func = (FindFunc) dlsym(library_handle, "operation_cache_find");
    contains_func = (ContainsFunc) dlsym(library_handle, "operation_cache_contains");
    size_func = (SizeFunc) dlsym(library_handle, "operation_cache_size");
    clear_func = (ClearFunc) dlsym(library_handle, "operation_cache_clear");
    
    dlerror(); // エラーをクリア
    
    cout << "Function pointers:" << endl;
    cout << "  graal_create_isolate: " << (void*)graal_create_isolate << endl;
    cout << "  graal_tear_down_isolate: " << (void*)graal_tear_down_isolate << endl;
    cout << "  insert_func: " << (void*)insert_func << endl;
    cout << "  find_func: " << (void*)find_func << endl;
    cout << "  contains_func: " << (void*)contains_func << endl;
    cout << "  size_func: " << (void*)size_func << endl;
    cout << "  clear_func: " << (void*)clear_func << endl;
    
    if (!graal_create_isolate || !insert_func || !find_func || !size_func) {
        dlclose(library_handle);
        throw runtime_error("Failed to load required functions from library");
    }
    
    // GraalVM Isolateを初期化
    if (graal_create_isolate) {
        cout << "Initializing GraalVM Isolate..." << endl;
        // int result = graal_create_isolate(&isolate_thread, &isolate);
        int result = graal_create_isolate(&isolate, &isolate_thread);
        if (result == 0) {
            graal_initialized = true;
            cout << "GraalVM Isolate initialized successfully" << endl;
        } else {
            cout << "Warning: GraalVM Isolate initialization failed with code: " << result << endl;
        }
    }
    
    cout << "OperationCacheClient: Successfully loaded library " << library_path << endl;
}

OperationCacheClient::~OperationCacheClient() {
    // GraalVM Isolateをクリーンアップ
    if (graal_initialized && graal_tear_down_isolate && isolate_thread) {
        cout << "Tearing down GraalVM Isolate..." << endl;
        graal_tear_down_isolate(isolate_thread);
    }
    
    if (library_handle) {
        dlclose(library_handle);
        cout << "OperationCacheClient: Library unloaded" << endl;
    }
}

void OperationCacheClient::initialize(const string& library_path) {
    if (!g_initialized) {
        g_library_path = library_path;
        g_initialized = true;
    }
}

// シングルトンインスタンス取得
OperationCacheClient& OperationCacheClient::getInstance() {
    static OperationCacheClient instance(g_library_path);
    return instance;
}


// ライブラリの再ロード（テスト用）
// void OperationCacheClient::reload(const string& library_path) {
//     lock_guard<mutex> lock(instance_mutex);
//     instance.reset();
//     instance = unique_ptr<OperationCacheClient>(new OperationCacheClient(library_path));
// }

// // インスタンスの解放
// void OperationCacheClient::shutdown() {
//     lock_guard<mutex> lock(instance_mutex);
//     instance.reset();
// }

// キャッシュ操作メソッド
void OperationCacheClient::insert(long long key, OperationResult result) {
    // if (!insert_func) {
    //     cerr << "Insert function not available" << endl;
    //     return;
    // }
    
    // int answer = insert_func(key, result.first.real(), result.first.imag(), result.second);
    // if (answer != 0) {
    //     cerr << "Insert failed with code: " << answer << endl;
    //     return;
    // }
    return;
}

OperationResult OperationCacheClient::find(long long key) {
    if (!find_func) {
        cerr << "Find function not available" << endl;
        return OperationResult();
    }

    char buffer[BUFFER_SIZE];
    memset(buffer, 0, BUFFER_SIZE);

    try {
        int result = find_func(key, buffer);
        
        if (result != 0) {
            if (result == -1) {
                // Not found (正常なケース)
                return OperationResult();
            } else {
                cerr << "Find failed with code: " << result << endl;
                return OperationResult();
            }
        }
        
        // JSONパース
        double real, imag;
        int64_t uniqueTableKey;
        int parsed = sscanf(buffer, "{\"real\":%lf,\"imag\":%lf,\"uniqueTableKey\":%lld}", 
                           &real, &imag, &uniqueTableKey);
        
        if (parsed == 3) {
            return OperationResult(complex<double>(real, imag), uniqueTableKey);
        } else {
            cerr << "Failed to parse result: " << buffer << endl;
            return OperationResult();
        }
    } catch (const std::exception& e) {
        cerr << "Exception in find: " << e.what() << endl;
        return OperationResult();
    } catch (...) {
        cerr << "Unknown exception in find" << endl;
        return OperationResult();
    }
}

bool OperationCacheClient::contains(long long key) {
    // if (!contains_func) {
    //     // fallback to find
    //     OperationResult result = find(key);
    //     return result != OperationResult();
    // }
    
    // int result = contains_func(key);
    // if (result < 0) {
    //     cerr << "Contains check failed with code: " << result << endl;
    //     return false;
    // }
    // return result == 1;
    return false;
}

int64_t OperationCacheClient::size() {
    if (!size_func) {
        cerr << "Size function not available" << endl;
        return -1;
    }
    
    if (!graal_initialized) {
        cerr << "GraalVM not initialized" << endl;
        return -1;
    }
    
    try {
        cout << "Calling size_func..." << endl;
        int64_t result = size_func();
        cout << "Size function returned: " << result << endl;
        return result;
    } catch (const std::exception& e) {
        cerr << "Exception in size: " << e.what() << endl;
        return -1;
    } catch (...) {
        cerr << "Unknown exception in size" << endl;
        return -1;
    }
}

bool OperationCacheClient::clear() {
    // if (!clear_func) {
    //     cerr << "Clear function not available" << endl;
    //     return false;
    // }
    
    // int result = clear_func();
    // if (result != 0) {
    //     cerr << "Clear failed with code: " << result << endl;
    //     return false;
    // }
    return true;
}

// デバッグ情報
void OperationCacheClient::printStatus() {
    cout << "=== OperationCacheClient Status ===" << endl;
    cout << "Library handle: " << (library_handle ? "loaded" : "not loaded") << endl;
    cout << "Insert function: " << (insert_func ? "available" : "not available") << endl;
    cout << "Find function: " << (find_func ? "available" : "not available") << endl;
    cout << "Contains function: " << (contains_func ? "available" : "not available") << endl;
    cout << "Size function: " << (size_func ? "available" : "not available") << endl;
    cout << "Clear function: " << (clear_func ? "available" : "not available") << endl;
    
    if (size_func) {
        int64_t current_size = size();
        cout << "Current cache size: " << current_size << endl;
    }
    cout << "=================================" << endl;
}