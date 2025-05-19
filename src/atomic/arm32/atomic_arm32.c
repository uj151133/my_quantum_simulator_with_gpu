#include "../atomic.h"

extern bool cas_arm32(void **ptr, void *expected, void *desired);

bool cas(void **ptr, void *expected, void *desired) {
    return cas_arm32(ptr, expected, desired);
}