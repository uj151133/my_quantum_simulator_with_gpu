    .globl _cas_arm64
_cas_arm64:
    mov     x3, #0
1:  ldaxr   x4, [x0]
    cmp     x4, x1
    b.ne    2f
    stlxr   w5, x2, [x0]
    cbnz    w5, 1b
    mov     x3, #1
2:  mov     x0, x3
    ret