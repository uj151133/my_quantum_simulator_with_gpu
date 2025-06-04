#ifndef __LIBOPERATION_CACHE_H
#define __LIBOPERATION_CACHE_H

#include <graal_isolate.h>


#if defined(__cplusplus)
extern "C" {
#endif

int operation_cache_insert(graal_isolatethread_t*, long long int, double, double, long long int);

int operation_cache_find(graal_isolatethread_t*, long long int, char*);

int operation_cache_contains(graal_isolatethread_t*, long long int);

long long int operation_cache_size(graal_isolatethread_t*);

int operation_cache_clear(graal_isolatethread_t*);

#if defined(__cplusplus)
}
#endif
#endif
