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
            jint res = g_jvm->GetEnv((void**)out_env, JNI_VERSION_1_8);
            if (res != JNI_OK) {
                *out_env = nullptr;
            }
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
    if (res != JNI_OK) return false;

    if (out_env) *out_env = env;

    // グローバル参照作成
    jclass localCacheCls = env->FindClass("OperationCache");
    if (!localCacheCls) return false;
    g_OperationCache_cls = (jclass)env->NewGlobalRef(localCacheCls);
    env->DeleteLocalRef(localCacheCls);

    if (!g_OperationCache_cls) return false;

    jclass localResultCls = env->FindClass("OperationResult");
    if (!localResultCls) return false;
    g_OperationResult_cls = (jclass)env->NewGlobalRef(localResultCls);
    env->DeleteLocalRef(localResultCls);

    if (!g_OperationResult_cls) return false;

    return true;
}

// スレッドごとのattach
void attachJni() {
    if (!g_jvm) return;
    if (!t_env) {
        JNIEnv* env = nullptr;
        jint res = g_jvm->GetEnv((void**)&env, JNI_VERSION_1_8);
        if (res == JNI_OK) {
            t_env = env;
        } else if (res == JNI_EDETACHED) {
            if (g_jvm->AttachCurrentThread((void**)&env, nullptr) == 0) {
                t_env = env;
            }
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

// Javaキャッシュinsert
void jniInsert(long long key, const std::complex<double>& value, long long size) {
    attachJni();

    if (!g_OperationCache_cls) return;

    jmethodID mid = t_env->GetStaticMethodID(g_OperationCache_cls, "doNativeInsert", "(JDDJ)V");
    if (!mid) return;

    t_env->CallStaticVoidMethod(g_OperationCache_cls, mid, (jlong)key, (jdouble)value.real(), (jdouble)value.imag(), (jlong)size);
}

OperationResult jniFind(long long key) {
    attachJni();
    OperationResult result;

    if (!g_OperationResult_cls) return result;

    jmethodID mid = t_env->GetStaticMethodID(g_OperationCache_cls, "doNativeFind", "(J)LOperationResult;");
    if (!mid) return result;

    jobject resultObj = t_env->CallStaticObjectMethod(g_OperationCache_cls, mid, (jlong)key);
    if (resultObj == nullptr) return result;

    jfieldID fidReal = t_env->GetFieldID(g_OperationResult_cls, "real", "D");
    jfieldID fidImag = t_env->GetFieldID(g_OperationResult_cls, "imag", "D");
    jfieldID fidSize = t_env->GetFieldID(g_OperationResult_cls, "size", "J");

    if (!fidReal || !fidImag || !fidSize) return result;

    double real = t_env->GetDoubleField(resultObj, fidReal);
    double imag = t_env->GetDoubleField(resultObj, fidImag);
    long long size = (long long)t_env->GetLongField(resultObj, fidSize);

    result.first = std::complex<double>(real, imag);
    result.second = size;

    t_env->DeleteLocalRef(resultObj);
    return result;
}