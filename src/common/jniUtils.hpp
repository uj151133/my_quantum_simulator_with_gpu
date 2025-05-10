#ifndef JNIUTILS_HPP
#define JNIUTILS_HPP
#define JNI_VERSION JNI_VERSION_1_8
#include "../models/qmdd.hpp"
#include <jni.h>
#include <complex>
#include <string>
#include <iostream>

// jniUtilsクラス
class jniUtils {
public:
    static jniUtils& getInstance();

    void jniInsert(long long key, const std::complex<double>& value, long long size);
    OperationResult jniFind(long long key);

private:
    jniUtils();
    ~jniUtils();
    jniUtils(const jniUtils&) = delete;
    jniUtils& operator=(const jniUtils&) = delete;

    // JNI環境
    JNIEnv* t_env = nullptr;
    static JavaVM* g_jvm;
    static jclass g_OperationCache_cls;
    static jclass g_OperationResult_cls;

    // JVMの初期化（必要なら呼び出す）
    static bool initJvm(const std::string& class_path, const std::string& caffeine_jar);
};

#endif