#include "jniUtils.hpp"

// グローバル変数の定義
JavaVM* jniUtils::g_jvm = nullptr;
jclass jniUtils::g_OperationCache_cls = nullptr;
jclass jniUtils::g_OperationResult_cls = nullptr;


jniUtils& jniUtils::getInstance() {
    thread_local jniUtils instance;
    return instance;
}

jniUtils::jniUtils() {
    // JVM初期化（初回のみ）
    if (!g_jvm) {
        JavaVMInitArgs vm_args;
        JavaVMOption options[1];
        std::string cp = "-Djava.class.path=./src/java:./src/java/caffeine-3.2.0.jar:.";
        options[0].optionString = (char*)cp.c_str();
        vm_args.version = JNI_VERSION;
        vm_args.nOptions = 1;
        vm_args.options = options;
        vm_args.ignoreUnrecognized = JNI_FALSE;

        JNIEnv* env = nullptr;
        if (JNI_CreateJavaVM(&g_jvm, (void**)&env, &vm_args) != JNI_OK) {
            throw std::runtime_error("JVM initialization failed");
        }

        // グローバル参照の作成
        g_OperationCache_cls = (jclass)env->NewGlobalRef(env->FindClass("OperationCache"));
        g_OperationResult_cls = (jclass)env->NewGlobalRef(env->FindClass("OperationResult"));
    }

    // スレッドローカルなJNIEnvの取得（アタッチ）
    JNIEnv* env = nullptr;
    if (g_jvm->GetEnv((void**)&env, JNI_VERSION) == JNI_EDETACHED) {
        if (g_jvm->AttachCurrentThread((void**)&env, nullptr) != 0) {
            throw std::runtime_error("Thread attach failed");
        }
    }
    t_env = env;
}

jniUtils::~jniUtils() {
    // スレッドデタッチ
    if (t_env) {
        g_jvm->DetachCurrentThread();
        t_env = nullptr;
    }
}

void jniUtils::jniInsert(long long key, const std::complex<double>& value, long long size) {
    jmethodID mid = t_env->GetStaticMethodID(g_OperationCache_cls, "doNativeInsert", "(JDDJ)V");
    t_env->CallStaticVoidMethod(g_OperationCache_cls, mid, (jlong)key, (jdouble)value.real(), (jdouble)value.imag(), (jlong)size);
}

OperationResult jniUtils::jniFind(long long key) {
    OperationResult result;
    jmethodID mid = t_env->GetStaticMethodID(g_OperationCache_cls, "doNativeFind", "(J)LOperationResult;");
    jobject resultObj = t_env->CallStaticObjectMethod(g_OperationCache_cls, mid, (jlong)key);
    if (resultObj) {
        jfieldID fidReal = t_env->GetFieldID(g_OperationResult_cls, "real", "D");
        jfieldID fidImag = t_env->GetFieldID(g_OperationResult_cls, "imag", "D");
        jfieldID fidSize = t_env->GetFieldID(g_OperationResult_cls, "size", "J");

        result.first = std::complex<double>(
            t_env->GetDoubleField(resultObj, fidReal),
            t_env->GetDoubleField(resultObj, fidImag)
        );
        result.second = (long long)t_env->GetLongField(resultObj, fidSize);
    }
    return result;
}