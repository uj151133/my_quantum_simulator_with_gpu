#ifndef JNIUTILS_HPP
#define JNIUTILS_HPP
#define JNI_VERSION JNI_VERSION_1_8
#include "../models/qmdd.hpp"
#include <jni.h>
#include <complex>
#include <string>
#include <iostream>
#include <mutex>

using namespace std;

// jniUtilsクラス
class jniUtils {
public:
    static jniUtils& getInstance();

    void jniInsert(long long key, const complex<double>& value, long long uniqueTableKey);
    OperationResult jniFind(long long key);

private:
    jniUtils();
    ~jniUtils();
    jniUtils(const jniUtils&) = delete;
    jniUtils& operator=(const jniUtils&) = delete;

    JNIEnv* getThreadEnv();

    static JavaVM* g_jvm;
    static jclass g_OperationCache_cls;
    static jclass g_OperationResult_cls;
    static mutex g_mutex;
    static bool g_initialized;

    static jmethodID g_doNativeInsert_mid;
    static jmethodID g_doNativeFind_mid;
    static jfieldID g_real_fid;
    static jfieldID g_imag_fid;
    static jfieldID g_uniqueTableKey_fid;

    static bool initJvm();
};

#endif