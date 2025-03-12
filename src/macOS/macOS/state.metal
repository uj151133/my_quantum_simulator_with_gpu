#pragma clang diagnostic ignored "-Wdouble-promotion"

#include <metal_stdlib>
#include "complex.metal"
#include "mathUtils.metal"
using namespace metal;

/////////////////////////////////////
//
//    KET VECTORS
//
/////////////////////////////////////

kernel void Ket0Vector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 1;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 1 && position.y < 2) {
        if (position.y == 0)  matrix.data[index] = ONE;
        else matrix.data[index] = ZERO;
    }
}

kernel void Ket1Vector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 1;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 1 && position.y < 2) {
        if (position.y == 0) matrix.data[index] = ZERO;
        else matrix.data[index] = ONE;
    }
}

kernel void KetPlusVector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 1;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 1 && position.y < 2) {
        matrix.data[index] = Complex(1.0f / sqrt(2.0f));
    }
}

kernel void KetMinus1Vector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 1;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 1 && position.y < 2) {
        if (position.y == 0) matrix.data[index] = Complex(1.0f / sqrt(2.0f));
        else matrix.data[index] = Complex(-1.0f / sqrt(2.0f));
    }
}

kernel void KetIVector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 1;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 1 && position.y < 2) {
        if (position.y == 0) matrix.data[index] = Complex(1.0f / sqrt(2.0f));
        else matrix.data[index] = i / sqrt(2.0f);
    }
}

kernel void KetIMinusVector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 1;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 1 && position.y < 2) {
        if (position.y == 0) matrix.data[index] = Complex(1.0f / sqrt(2.0f));
        else matrix.data[index] = -i / sqrt(2.0f);
    }
}

/////////////////////////////////////
//
//    BRA VECTORS
//
/////////////////////////////////////

kernel void Bra0Vector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 1;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 1) {
        if (position.x == 0)  matrix.data[index] = ONE;
        else matrix.data[index] = ZERO;
    }
}

kernel void Bra1Vector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 1;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 1) {
        if (position.x == 0)  matrix.data[index] = ZERO;
        else matrix.data[index] = ONE;
    }
}

kernel void BraPlusVector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 1;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 1) {
        matrix.data[index] = Complex(1.0f / sqrt(2.0f));
    }
}

kernel void BraMinusVector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 1;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 1) {
        if (position.x == 0)  matrix.data[index] = Complex(1.0f / sqrt(2.0f));
        else matrix.data[index] = Complex(-1.0f / sqrt(2.0f));
    }
}

kernel void BraIVector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 1;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 1) {
        if (position.x == 0)  matrix.data[index] = Complex(1.0f / sqrt(2.0f));
        else matrix.data[index] = i / sqrt(2.0f);
    }
}

kernel void BraIMinusVector(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 1;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 1) {
        if (position.x == 0)  matrix.data[index] = Complex(1.0f / sqrt(2.0f));
        else matrix.data[index] = -i / sqrt(2.0f);
    }
}
