#ifndef ATOMIC_H
#define ATOMIC_H

#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>
// #include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

bool cas(void **ptr, void *expected, void *desired);

// // x86_64用CAS
// bool cas_x86_64(int *ptr, int expected, int desired);

// // ARM64用LL/SCインクリメント
// void atomic_increment_arm64(int *ptr);
// // ARM64用CAS
// bool cas_arm64(int *ptr, int expected, int desired);

// // RISC-V用LL/SCインクリメント
// void atomic_increment_riscv(int *ptr);
// // RISC-V用CAS
// bool cas_riscv(int *ptr, int expected, int desired);

// // MIPS用LL/SCインクリメント
// void atomic_increment_mips(int *ptr);
// // MIPS用CAS
// bool cas_mips(int *ptr, int expected, int desired);

// // PowerPC用LL/SCインクリメント
// void atomic_increment_powerpc(int *ptr);
// // PowerPC用CAS
// bool cas_powerpc(int *ptr, int expected, int desired);

#ifdef __cplusplus
}
#endif

#endif