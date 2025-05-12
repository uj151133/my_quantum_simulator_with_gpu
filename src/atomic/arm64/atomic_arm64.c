// #include "../atomic_ops.h"

// #if defined(__aarch64__)

// void atomic_increment_arm64(int *ptr) {
//     int tmp, result;
//     do {
//         __asm__ __volatile__(
//             "ldxr   %w0, [%2]\n"
//             "add    %w0, %w0, #1\n"
//             "stxr   %w1, %w0, [%2]\n"
//             : "=&r"(tmp), "=&r"(result)
//             : "r"(ptr)
//             : "memory"
//         );
//     } while (result != 0);
// }

// bool cas_arm64(int *ptr, int expected, int desired) {
//     int tmp;
//     unsigned int result;
//     bool success;
//     __asm__ __volatile__(
//         "1: ldxr    %w0, [%3]\n"
//         "   cmp     %w0, %w4\n"
//         "   b.ne    2f\n"
//         "   stxr    %w1, %w5, [%3]\n"
//         "   cbnz    %w1, 1b\n"
//         "   mov     %w2, #1\n"
//         "   b       3f\n"
//         "2: mov     %w2, #0\n"
//         "3:"
//         : "=&r"(tmp), "=&r"(result), "=&r"(success)
//         : "r"(ptr), "r"(expected), "r"(desired)
//         : "memory", "cc"
//     );
//     return success;
// }

// #else
// void atomic_increment_arm64(int *ptr) { (void)ptr; }
// bool cas_arm64(int *ptr, int expected, int desired) {
//     (void)ptr; (void)expected; (void)desired; return false;
// }
// #endif

#include "../atomic.h"

// ARM64/AArch64専用
bool cas_arm64(void **ptr, void *expected, void *desired) {
    void *old;
    int success;
    __asm__ __volatile__(
        "1: ldxr %0, [%2]\n"            // old = *ptr
        "   cmp  %0, %3\n"              // old == expected ?
        "   b.ne 2f\n"                  // 違えば失敗
        "   stxr %w1, %4, [%2]\n"       // *ptr = desired (if not contended), w1 = 0(success)/1(fail)
        "   cbnz %w1, 1b\n"             // 失敗ならリトライ
        "2:"
        : "=&r"(old), "=&r"(success)
        : "r"(ptr), "r"(expected), "r"(desired)
        : "memory"
    );
    return (old == expected) && (success == 0);
}