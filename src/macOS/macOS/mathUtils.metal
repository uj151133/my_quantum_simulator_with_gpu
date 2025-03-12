#pragma clang diagnostic ignored "-Wdouble-promotion"

#include <metal_stdlib>
#include "complex.metal"
using namespace metal;

inline Complex complexExp(Complex z) {
    return Complex(exp(z.value.x) * cos(z.value.y), exp(z.value.x) * sin(z.value.y));
}

inline float complexAbs(Complex z) {
    return sqrt(pow(z.value.x, 2) + pow(z.value.y, 2));
}

