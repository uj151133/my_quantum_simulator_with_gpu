#pragma once
#include <jni.h>
#include <complex>
#include "../models/qmdd.hpp"

using namespace std;


extern JavaVM* g_jvm;
extern thread_local JNIEnv* t_env;
extern jclass g_OperationCache_cls;

// JVM初期化
bool initJvm(const std::string& class_path, const std::string& caffeine_jar, JNIEnv** out_env = nullptr);

// スレッドごとのattach/detach
void attachJni();
void detachJni();

// プール全スレッドでattach/detach
void attachJniForAllThreads();
void detachJniForAllThreads();

// Javaキャッシュinsert/find
void jniInsert(long long key, const std::complex<double>& value, long long size);
OperationResult jniFind(long long key);