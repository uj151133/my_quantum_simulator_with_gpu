    .globl _cas_x86_64
_cas_x86_64:
    movq    %rsi, %rax
    xorl    %ecx, %ecx
    lock cmpxchgq    %rdx, (%rdi)
    sete    %cl
    movl    %ecx, %eax
    retq