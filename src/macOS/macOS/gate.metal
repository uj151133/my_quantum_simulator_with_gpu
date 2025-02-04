#include <metal_stdlib>
#include "base.metal"
#include "mathUtils.metal"
using namespace metal;

kernel void makeZEROMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        matrix.data[index] = ZERO;
    }
}

kernel void makeIdentityMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                               uint2 position [[thread_position_in_grid]],
                               uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        matrix.data[index] = (position.x == position.y) ? ONE : ZERO;
    }
}

kernel void makePhMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                         constant float* delta [[buffer(1)]],
                         uint2 position [[thread_position_in_grid]],
                         uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        matrix.data[index] = (position.x == position.y) ? complexExp(i * *delta) : ZERO;
    }
}

kernel void makeXMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                               uint2 position [[thread_position_in_grid]],
                               uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        matrix.data[index] = (position.x == position.y) ? ZERO : ONE;
    }
}

kernel void makeYMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        if (position.x != position.y) {
            matrix.data[index] = (position.y < position.x) ? -i : i;
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void makeZMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        if (position.x == position.y) {
            matrix.data[index] = (position.x == 0) ? ONE : -ONE;
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void makeSMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        if (position.x == position.y) {
            matrix.data[index] = (position.x == 0) ? ONE : i;
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void makeSDaggerMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        if (position.x == position.y) {
            matrix.data[index] = (position.x == 0) ? ONE : -i;
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void makeVMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                               uint2 position [[thread_position_in_grid]],
                               uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        float inv2 = 1.0f / 2.0f;
        matrix.data[index] = (position.x == position.y) ? Complex(inv2, inv2) : Complex(inv2, -inv2);
    }
}

kernel void makeVDaggerMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                               uint2 position [[thread_position_in_grid]],
                               uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        float inv2 = 1.0f / 2.0f;
        matrix.data[index] = (position.x == position.y) ? Complex(inv2, -inv2) : Complex(inv2, inv2);
    }
}

kernel void makeHMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        float invSqrt2 = 1.0f / sqrt(2.0f);
        matrix.data[index] = (position.y == 1 && position.x == 1) ? Complex(-invSqrt2) : Complex(invSqrt2);
    }
}

kernel void makeCXMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        if ((position.y == 0 && position.x == 0) ||
            (position.y == 1 && position.x == 1) ||
            (position.y == 2 && position.x == 3) ||
            (position.y == 3 && position.x == 2)) {
            matrix.data[index] = ONE;
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void makePMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        constant float* phi [[buffer(1)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        if (position.x == position.y) {
            matrix.data[index] = (position.x == 0) ? ONE : complexExp(i * *phi);
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void makeTMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        if (position.x == position.y) {
            matrix.data[index] = (position.x == 0) ? ONE : complexExp(i * PI / 4.0f);
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void makeTDaggerMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        if (position.x == position.y) {
            matrix.data[index] = (position.x == 0) ? ONE : complexExp(-i * PI / 4.0f);
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void makeRxMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                         constant float* theta [[buffer(1)]],
                         uint2 position [[thread_position_in_grid]],
                         uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        float thetaHalf = *theta / 2.0f;
        matrix.data[index] = (position.x == position.y) ? Complex(cos(thetaHalf)) :  -i * sin(thetaHalf);
    }
}

kernel void makeRyMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                         constant float* theta [[buffer(1)]],
                         uint2 position [[thread_position_in_grid]],
                         uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        float thetaHalf = *theta / 2.0f;
        if (position.x != position.y) {
            matrix.data[index] = (position.y < position.x) ? Complex(-sin(thetaHalf)) : Complex(sin(thetaHalf));
        } else {
            matrix.data[index] = Complex(cos(thetaHalf));
        }
    }
}

kernel void makeRzMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                         constant float* theta [[buffer(1)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        float thetaHalf = *theta / 2.0f;
        if (position.x == position.y) {
            matrix.data[index] = (position.x == 0) ? complexExp(-i * thetaHalf) : complexExp(i * thetaHalf);
        } else {
            matrix.data[index] = ZERO;
        }
    }
}
