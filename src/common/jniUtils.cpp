#include "jniUtils.hpp"

// グローバル変数の定義
JavaVM* jniUtils::g_jvm = nullptr;
jclass jniUtils::g_OperationCache_cls = nullptr;
jclass jniUtils::g_OperationResult_cls = nullptr;
mutex jniUtils::g_mutex;
bool jniUtils::g_initialized = false;

jmethodID jniUtils::g_doNativeInsert_mid = nullptr;
jmethodID jniUtils::g_doNativeFind_mid = nullptr;
jfieldID jniUtils::g_real_fid = nullptr;
jfieldID jniUtils::g_imag_fid = nullptr;
jfieldID jniUtils::g_uniqueTableKey_fid = nullptr;

jniUtils& jniUtils::getInstance() {
    static jniUtils instance;
    return instance;
}

jniUtils::jniUtils() {
    lock_guard<mutex> lock(this->g_mutex);

    if (!g_initialized) {
        if (!this->initJvm()) {
            throw runtime_error("JVM initialization failed");
        }
        this->g_initialized = true;
    }
}

bool jniUtils::initJvm() {
    JavaVMInitArgs vm_args;
    JavaVMOption options[1];
    string cp = "-Djava.class.path=./src/java:./src/java/caffeine-3.2.0.jar:.";
    options[0].optionString = (char*)cp.c_str();
    vm_args.version = JNI_VERSION;
    vm_args.nOptions = 1;
    vm_args.options = options;
    vm_args.ignoreUnrecognized = JNI_FALSE;

    JNIEnv* env = nullptr;
    if (JNI_CreateJavaVM(&g_jvm, (void**)&env, &vm_args) != JNI_OK) {
        return false;
    }

    g_OperationCache_cls = (jclass)env->NewGlobalRef(env->FindClass("OperationCache"));
    g_OperationResult_cls = (jclass)env->NewGlobalRef(env->FindClass("OperationResult"));
    
    if (!g_OperationCache_cls || !g_OperationResult_cls) {
        return false;
    }

    g_doNativeInsert_mid = env->GetStaticMethodID(g_OperationCache_cls, "doNativeInsert", "(JDDJ)V");
    g_doNativeFind_mid = env->GetStaticMethodID(g_OperationCache_cls, "doNativeFind", "(J)LOperationResult;");

    g_real_fid = env->GetFieldID(g_OperationResult_cls, "real", "D");
    g_imag_fid = env->GetFieldID(g_OperationResult_cls, "imag", "D");
    g_uniqueTableKey_fid = env->GetFieldID(g_OperationResult_cls, "uniqueTableKey", "J");

    if (!g_doNativeInsert_mid || !g_doNativeFind_mid || 
        !g_real_fid || !g_imag_fid || !g_uniqueTableKey_fid) {
        return false;
    }
    return true;
}

JNIEnv* jniUtils::getThreadEnv() {
    JNIEnv* env = nullptr;
    jint result = this->g_jvm->GetEnv((void**)&env, JNI_VERSION);
    
    if (result == JNI_EDETACHED) {
        if (this->g_jvm->AttachCurrentThread((void**)&env, nullptr) != JNI_OK) {
            throw runtime_error("Thread attach failed");
        }
    } else if (result != JNI_OK) {
        throw runtime_error("Failed to get JNI environment");
    }
    
    return env;
}

jniUtils::~jniUtils() {}

void jniUtils::jniInsert(long long key, const std::complex<double>& value, long long uniqueTableKey) {
    JNIEnv* env = this->getThreadEnv();
    env->CallStaticVoidMethod(this->g_OperationCache_cls, this->g_doNativeInsert_mid, (jlong)key, (jdouble)value.real(), (jdouble)value.imag(), (jlong)uniqueTableKey);
}

OperationResult jniUtils::jniFind(long long key) {
    JNIEnv* env = this->getThreadEnv();
    OperationResult result;
    jobject resultObj = env->CallStaticObjectMethod(this->g_OperationCache_cls, this->g_doNativeFind_mid, (jlong)key);
    if (resultObj) {

        result.first = complex<double>(
            env->GetDoubleField(resultObj, this->g_real_fid),
            env->GetDoubleField(resultObj, this->g_imag_fid)
        );
        result.second = (long long)env->GetLongField(resultObj, this->g_uniqueTableKey_fid);
    }
    return result;
}