#ifndef CAFFEINEBRIDGE_HPP
#define CAFFEINEBRIDGE_HPP

#include <jni.h>
#include "../models/qmdd.hpp"

class CaffeineBridge {
public:
    static void initJVM();
    static void write(size_t key, OperationResult result);
    static OperationResult read(size_t key);
    static void shutdownJVM();

private:
    static JavaVM* jvm;
    static JNIEnv* env;
    static jclass cacheClass;
};

#endif
