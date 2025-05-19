#include "../atomic.h"

extern bool cas_x86(void **ptr, void *expected, void *desired);

bool cas(void **ptr, void *expected, void *desired) {
    return cas_x86(ptr, expected, desired);
}