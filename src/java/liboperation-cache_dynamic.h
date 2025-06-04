#ifndef __LIBOPERATION_CACHE_H
#define __LIBOPERATION_CACHE_H

#include <graal_isolate_dynamic.h>


#if defined(__cplusplus)
extern "C" {
#endif

typedef int (*operation_cache_insert_fn_t)(graal_isolatethread_t*, long long int, double, double, long long int);

typedef int (*operation_cache_find_fn_t)(graal_isolatethread_t*, long long int, char*);

typedef int (*operation_cache_contains_fn_t)(graal_isolatethread_t*, long long int);

typedef long long int (*operation_cache_size_fn_t)(graal_isolatethread_t*);

typedef int (*operation_cache_clear_fn_t)(graal_isolatethread_t*);

#if defined(__cplusplus)
}
#endif
#endif
