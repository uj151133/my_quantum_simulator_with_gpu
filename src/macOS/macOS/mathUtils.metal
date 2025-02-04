#include <metal_stdlib>
#include "base.metal"
using namespace metal;

inline Complex complexExp(Complex z) {
    float realPart = exp(z.real) * cos(z.imag);
    float imagPart = exp(z.real) * sin(z.imag);
    return Complex(realPart, imagPart);
}


