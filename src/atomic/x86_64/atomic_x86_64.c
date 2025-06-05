#include "../atomic.h"

extern bool cas_x86_64(void **ptr, void *expected, void *desired);

bool cas(void **ptr, void *expected, void *desired) {
    return cas_x86_64(ptr, expected, desired);
}