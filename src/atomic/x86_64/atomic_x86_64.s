.section .text
    .globl cas_x86_64
cas_x86_64:
    movq    %rsi, %rax
    xorl    %ecx, %ecx
    lock cmpxchgq    %rdx, (%rdi)
    sete    %cl
    movl    %ecx, %eax
    retq

.section .note.GNU-stack,"",@progbits