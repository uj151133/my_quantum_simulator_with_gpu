#include "../atomic.h"

#ifdef __cplusplus
extern "C" {
#endif

extern bool cas_arm64(void **ptr, void *expected, void *desired);

#ifdef __cplusplus
}
#endif

bool cas(void **ptr, void *expected, void *desired) {
    return cas_arm64(ptr, expected, desired);
}