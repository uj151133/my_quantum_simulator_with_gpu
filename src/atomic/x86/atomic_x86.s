    .globl _cas_x86_32
_cas_x86_32:
    movl    8(%esp), %eax
    xorl    %ecx, %ecx
    lock cmpxchgl 12(%esp), (%esp)
    sete    %cl
    movl    %ecx, %eax
    ret