#include "CaffeineBridge.hpp"
#include <iostream>

JavaVM* CaffeineBridge::jvm = nullptr;
JNIEnv* CaffeineBridge::env = nullptr;
jclass CaffeineBridge::cacheClass = nullptr;

void CaffeineBridge::initJVM() {
    JavaVMInitArgs vm_args;
    JavaVMOption options[1];
    options[0].optionString = const_cast<char*>("-Djava.class.path=.:caffeine-3.2.0.jar");
    vm_args.version = JNI_VERSION_1_8;
    vm_args.nOptions = 1;
    vm_args.options = options;
    vm_args.ignoreUnrecognized = false;

    jint res = JNI_CreateJavaVM(&jvm, (void**)&env, &vm_args);
    if (res != JNI_OK) {
        std::cerr << "❌ JVM 起動失敗" << std::endl;
        exit(1);
    }

    cacheClass = env->FindClass("OperationCache");
    if (!cacheClass) {
        std::cerr << "❌ OperationCache クラスが見つかりません" << std::endl;
        exit(1);
    }
}

void CaffeineBridge::write(size_t key, OperationResult result) {
    jlong jKey = static_cast<jlong>(key);
    double real = result.first.real();
    double imag = result.first.imag();
    jlong index = static_cast<jlong>(result.second);
    jmethodID putMethod = env->GetStaticMethodID(
        cacheClass, "putFromCpp", "(JDDJ)V");
    env->CallStaticVoidMethod(cacheClass, putMethod, jKey, real, imag, index);
}

OperationResult CaffeineBridge::read(size_t key) {
    jlong jKey = static_cast<jlong>(key);
    jmethodID getMethod = env->GetStaticMethodID(
        cacheClass, "getFromCpp", "(J)LOperationResult;");
    jobject resultObj = env->CallStaticObjectMethod(cacheClass, getMethod, jKey);

    if (!resultObj) {
        std::cout << "❌ 見つかりませんでした: key=" << jKey << std::endl;
        return OperationResult();
    }

    jclass resultClass = env->GetObjectClass(resultObj);
    jfieldID realField = env->GetFieldID(resultClass, "real", "D");
    jfieldID imagField = env->GetFieldID(resultClass, "imag", "D");
    jfieldID indexField = env->GetFieldID(resultClass, "index", "J");

    double r = env->GetDoubleField(resultObj, realField);
    double i = env->GetDoubleField(resultObj, imagField);
    long idx = env->GetLongField(resultObj, indexField);

    std::cout << "✅ 読み込み成功: real=" << r << ", imag=" << i << ", index=" << idx << std::endl;
    return OperationResult(complex<double>(r, i), static_cast<size_t>(idx));
}

void CaffeineBridge::shutdownJVM() {
    if (jvm) jvm->DestroyJavaVM();
}
