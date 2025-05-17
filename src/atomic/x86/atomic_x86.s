    .globl cas_x86
cas_x86:
    movl    8(%esp), %eax
    xorl    %ecx, %ecx
    movl    4(%esp), %edx
    movl    12(%esp), %ebx
    lock cmpxchgl %ebx, (%edx)
    sete    %cl
    movl    %ecx, %eax
    ret