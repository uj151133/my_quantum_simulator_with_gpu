    .global _cas_arm32
_cas_arm32:
    mov     r3, #0
1:  ldrex   r4, [r0]
    cmp     r4, r1
    bne     2f
    strex   r5, r2, [r0]
    cmp     r5, #0
    bne     1b
    mov     r3, #1
2:  mov     r0, r3
    bx      lr