#include "jniUtils.hpp"
#include "../modules/threadPool.hpp"
#include <boost/asio/post.hpp>
#include <thread>
#include <atomic>
#include <iostream>
#include <cassert>

JavaVM* g_jvm = nullptr;
thread_local JNIEnv* t_env = nullptr;
jclass g_OperationCache_cls = nullptr;
jclass g_OperationResult_cls = nullptr;

// JVM起動
bool initJvm(const std::string& class_path, const std::string& caffeine_jar, JNIEnv** out_env) {
    if (g_jvm) {
        if (out_env) {
            // 既に起動済みの場合
            jint res = g_jvm->GetEnv((void**)out_env, JNI_VERSION_1_8);
            if (res != JNI_OK) *out_env = nullptr;
        }
        return true;
    }
    JavaVMInitArgs vm_args;
    JavaVMOption options[1];
    std::string cp = "-Djava.class.path=" + caffeine_jar + ":" + class_path + ":.";
    options[0].optionString = (char*)cp.c_str();
    vm_args.version = JNI_VERSION_1_8;
    vm_args.nOptions = 1;
    vm_args.options = options;
    vm_args.ignoreUnrecognized = JNI_FALSE;
    JNIEnv* env = nullptr;
    jint res = JNI_CreateJavaVM(&g_jvm, (void**)&env, &vm_args);
    if (out_env) *out_env = env;
    if (res != JNI_OK) return false;

    // JVM起動直後にグローバル参照作成
    jclass localCacheCls = env->FindClass("OperationCache");
    // if (!localCls) {
    //     // std::cerr << "[JNI DEBUG] FindClass OperationCache failed!" << std::endl;
    //     env->ExceptionDescribe();
    //     return false;
    // }
    g_OperationCache_cls = (jclass)env->NewGlobalRef(localCacheCls);
    env->DeleteLocalRef(localCacheCls);
    jclass localResultCls = env->FindClass("OperationResult");
    g_OperationResult_cls = (jclass)env->NewGlobalRef(localResultCls);
    env->DeleteLocalRef(localResultCls);
    return true;
}

// スレッドごとのattach
void attachJni() {
    if (!g_jvm) return;
    if (!t_env) {
        JNIEnv* env = nullptr;
        jint res = g_jvm->GetEnv((void**)&env, JNI_VERSION_1_8);
        if (res == JNI_OK) t_env = env;
        else if (res == JNI_EDETACHED) {
            if (g_jvm->AttachCurrentThread((void**)&env, nullptr) == 0)
                t_env = env;
        }
    }
}

// スレッドごとのdetach
void detachJni() {
    if (g_jvm && t_env) {
        g_jvm->DetachCurrentThread();
        t_env = nullptr;
    }
}

// プール全スレッドでattach
void attachJniForAllThreads() {
    std::atomic<int> attached{0};
    for (int i = 0; i < std::thread::hardware_concurrency(); ++i) {
        boost::asio::post(threadPool, [&attached]() {
            attachJni();
            ++attached;
        });
    }
    while (attached.load() < std::thread::hardware_concurrency()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// detachも同様
void detachJniForAllThreads() {
    std::atomic<int> detached{0};
    for (int i = 0; i < std::thread::hardware_concurrency(); ++i) {
        boost::asio::post(threadPool, [&detached]() {
            detachJni();
            ++detached;
        });
    }
    while (detached.load() < std::thread::hardware_concurrency()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// Javaキャッシュinsert
void jniInsert(long long key, const std::complex<double>& value, long long size) {
    jmethodID mid = t_env->GetStaticMethodID(g_OperationCache_cls, "doNativeInsert", "(JDDJ)V");
    t_env->CallStaticVoidMethod(g_OperationCache_cls, mid, (jlong)key, (jdouble)value.real(), (jdouble)value.imag(), (jlong)size);
}

// Javaキャッシュfind
OperationResult jniFind(long long key) {
    attachJni();
    OperationResult result;
    jmethodID mid = t_env->GetStaticMethodID(g_OperationCache_cls, "doNativeFind", "(J)LOperationResult;");
    jobject resultObj = t_env->CallStaticObjectMethod(g_OperationCache_cls, mid, (jlong)key);
    if (resultObj == nullptr) {
        return result;
    }else {
        jfieldID fidReal = t_env->GetFieldID(g_OperationResult_cls, "real", "D");
        jfieldID fidImag = t_env->GetFieldID(g_OperationResult_cls, "imag", "D");
        jfieldID fidUniqueTableKey = t_env->GetFieldID(g_OperationResult_cls, "size", "J");

        double real = t_env->GetDoubleField(resultObj, fidReal);
        double imag = t_env->GetDoubleField(resultObj, fidImag);
        long long uniqueTableKey = (long long)t_env->GetLongField(resultObj, fidUniqueTableKey);
        result.first = complex<double>(real, imag);
        result.second = uniqueTableKey;
        return result;
    }
}