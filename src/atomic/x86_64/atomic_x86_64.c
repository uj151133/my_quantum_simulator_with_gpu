#include "../atomic.h"

extern bool cas_x86_64(void **ptr, void *expected, void *desired);

bool cas(void **ptr, void *expected, void *desired) {
    if (expected == NULL || ptr == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to cas_x86_64\n");
        fprintf(stderr, "ptr: %p, expected: %p, desired: %p\n", ptr, expected, desired);
        return false;
    }
    return cas_x86_64(ptr, expected, desired);
}