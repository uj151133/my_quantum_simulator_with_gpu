#include <metal_stdlib>
using namespace metal;

struct Complex {
    double real;
    double imag;
};

struct QMDDGate {
    Complex matrix[2][2];
};

kernel void identityGate(device QMDDGate* result [[buffer(0)]]) {
    result->matrix[0][0] = Complex{1.0, 0.0};
    result->matrix[0][1] = Complex{0.0, 0.0};
    result->matrix[1][0] = Complex{0.0, 0.0};
    result->matrix[1][1] = Complex{1.0, 0.0};
}

kernel void phaseGate(device QMDDGate* result [[buffer(0)]], constant double& delta [[buffer(1)]]) {
    double cosDelta = cos(delta);
    double sinDelta = sin(delta);
    result->matrix[0][0] = Complex{1.0, 0.0};
    result->matrix[0][1] = Complex{0.0, 0.0};
    result->matrix[1][0] = Complex{0.0, 0.0};
    result->matrix[1][1] = Complex{cosDelta, sinDelta};
}