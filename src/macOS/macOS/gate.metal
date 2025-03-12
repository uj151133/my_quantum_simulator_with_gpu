#pragma clang diagnostic ignored "-Wdouble-promotion"

#include <metal_stdlib>
#include "complex.metal"
#include "mathUtils.metal"
using namespace metal;

kernel void ZEROMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                           uint2 position [[thread_position_in_grid]],
                           uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        matrix.data[index] = ZERO;
    }
}

kernel void IdentityMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                               uint2 position [[thread_position_in_grid]],
                               uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        matrix.data[index] = (position.x == position.y) ? ONE : ZERO;
    }
}

kernel void PhMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void XMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                               uint2 position [[thread_position_in_grid]],
                               uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        matrix.data[index] = (position.x == position.y) ? ZERO : ONE;
    }
}

kernel void YMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void ZMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void SMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void SDaggerMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void VMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void VDaggerMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void HMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void CX1Matrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void CX2Matrix(device ComplexMatrix& matrix [[buffer(0)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        if ((position.y == 0 && position.x == 0) ||
            (position.y == 1 && position.x == 3) ||
            (position.y == 2 && position.x == 2) ||
            (position.y == 3 && position.x == 1)) {
            matrix.data[index] = ONE;
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void varCXMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        if ((position.y == 0 && position.x == 1) ||
            (position.y == 1 && position.x == 0) ||
            (position.y == 2 && position.x == 2) ||
            (position.y == 3 && position.x == 3)) {
            matrix.data[index] = ONE;
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void CZMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        if ((position.y == 0 && position.x == 0) ||
            (position.y == 1 && position.x == 1) ||
            (position.y == 2 && position.x == 2)) {
            matrix.data[index] = ONE;
        }else if (position.y == 3 && position.x == 3) {
            matrix.data[index] = -ONE;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void DCNOTMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        if ((position.y == 0 && position.x == 0) ||
            (position.y == 1 && position.x == 2) ||
            (position.y == 2 && position.x == 3) ||
            (position.y == 3 && position.x == 1)) {
            matrix.data[index] = ONE;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void SWAPMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        if ((position.y == 0 && position.x == 0) ||
            (position.y == 1 && position.x == 2) ||
            (position.y == 2 && position.x == 1) ||
            (position.y == 3 && position.x == 3)) {
            matrix.data[index] = ONE;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void iSWAPMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        if ((position.y == 0 && position.x == 0) ||
            (position.y == 3 && position.x == 3)) {
            matrix.data[index] = ONE;
        }else if ((position.y == 1 && position.x == 2) ||
                  (position.y == 2 && position.x == 1)){
            matrix.data[index] = i;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void PMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void TMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        if (position.x == position.y) {
            matrix.data[index] = (position.x == 0) ? ONE : complexExp(i * M_PI_F / 4.0f);
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void TDaggerMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    if (position.x < 2 && position.y < 2) {
        if (position.x == position.y) {
            matrix.data[index] = (position.x == 0) ? ONE : complexExp(-i * M_PI_F / 4.0f);
        } else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void CPMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        constant float* phi [[buffer(1)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        if ((position.y == 0 && position.x == 0) ||
            (position.y == 1 && position.x == 1) ||
            (position.y == 2 && position.x == 2)) {
            matrix.data[index] = ONE;
        }else if (position.y == 3 && position.x == 3) {
            matrix.data[index] = complexExp(i * *phi);
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void CSMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        if ((position.y == 0 && position.x == 0) ||
            (position.y == 1 && position.x == 1) ||
            (position.y == 2 && position.x == 2)) {
            matrix.data[index] = ONE;
        }else if (position.y == 3 && position.x == 3) {
            matrix.data[index] = i;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void RxMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void RyMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void RzMatrix(device ComplexMatrix& matrix [[buffer(0)]],
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

kernel void RxxMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        constant float* theta [[buffer(1)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        float thetaHalf = *theta / 2.0f;
        if (position.y == position.x) {
            matrix.data[index] = cos(thetaHalf);
        }else if (position.y + position.x == 3){
            matrix.data[index] = -i * sin(thetaHalf);
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void RyyMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        constant float* theta [[buffer(1)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        float thetaHalf = *theta / 2.0f;
        if (position.y == position.x) {
            matrix.data[index] = cos(thetaHalf);
        }else if((position.y == 0 && position.x == 3) ||
                 (position.y == 3 && position.x == 0)){
            matrix.data[index] = i * sin(thetaHalf);
        }else if ((position.y == 1 && position.x == 2) ||
                  (position.y == 2 && position.x == 1)){
            matrix.data[index] = -i * sin(thetaHalf);
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void RzzMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        constant float* theta [[buffer(1)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        float thetaHalf = *theta / 2.0f;
        if((position.y == 1 && position.x == 1) ||
           (position.y == 2 && position.x == 2)){
            matrix.data[index] = complexExp(i * thetaHalf);
        }else if ((position.y == 0 && position.x == 0) ||
                  (position.y == 3 && position.x == 3)){
            matrix.data[index] = complexExp(-i * thetaHalf);
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void RxyMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        constant float* theta [[buffer(1)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        float thetaHalf = *theta / 2.0f;
        if((position.y == 1 && position.x == 1) ||
           (position.y == 2 && position.x == 2)){
            matrix.data[index] = cos(thetaHalf);
        }else if ((position.y == 1 && position.x == 2) ||
                  (position.y == 2 && position.x == 1)){
            matrix.data[index] = -i * sin(thetaHalf);
        }else if ((position.y == 0 && position.x == 0) ||
                  (position.y == 3 && position.x == 3)){
            matrix.data[index] = ONE;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void SqureSWAPMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
    
        if((position.y == 1 && position.x == 1) ||
           (position.y == 2 && position.x == 2)){
            matrix.data[index] = Complex(1.0f / 2.0f, 1.0f / 2.0f);
        }else if ((position.y == 1 && position.x == 2) ||
                  (position.y == 2 && position.x == 1)){
            matrix.data[index] = Complex(1.0f / 2.0f, -1.0f / 2.0f);
        }else if ((position.y == 0 && position.x == 0) ||
                  (position.y == 3 && position.x == 3)){
            matrix.data[index] = ONE;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void SqureiSWAPMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                        uint2 position [[thread_position_in_grid]],
                        uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
    
        if((position.y == 1 && position.x == 1) ||
           (position.y == 2 && position.x == 2)){
            matrix.data[index] = Complex(1.0f / sqrt(2.0f));
        }else if ((position.y == 1 && position.x == 2) ||
                  (position.y == 2 && position.x == 1)){
            matrix.data[index] = i / sqrt(2.0f);
        }else if ((position.y == 0 && position.x == 0) ||
                  (position.y == 3 && position.x == 3)){
            matrix.data[index] = ONE;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void SWAPAlphaMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                                 constant float* alpha [[buffer(1)]],
                                 uint2 position [[thread_position_in_grid]],
                                 uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
    
        if((position.y == 1 && position.x == 1) ||
           (position.y == 2 && position.x == 2)){
            matrix.data[index] = (1.0f + complexExp(i * M_PI_F * *alpha)) / 2.0f;
        }else if ((position.y == 1 && position.x == 2) ||
                  (position.y == 2 && position.x == 1)){
            matrix.data[index] = (1.0f - complexExp(i * M_PI_F * *alpha)) / 2.0f;
        }else if ((position.y == 0 && position.x == 0) ||
                  (position.y == 3 && position.x == 3)){
            matrix.data[index] = ONE;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void FREDKINMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                          uint2 position [[thread_position_in_grid]],
                          uint2 threads [[threads_per_grid]]) {
    matrix.cols = 8;
    matrix.rows = 8;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 8 && position.y < 8) {
    
        if ((position.y == 0 && position.x == 0) ||
            (position.y == 1 && position.x == 1) ||
            (position.y == 2 && position.x == 2) ||
            (position.y == 3 && position.x == 3) ||
            (position.y == 4 && position.x == 4) ||
            (position.y == 5 && position.x == 6) ||
            (position.y == 6 && position.x == 5) ||
            (position.y == 7 && position.x == 7)) {
            matrix.data[index] = ONE;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}

kernel void UMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                    constant float* theta [[buffer(1)]],
                    constant float* phi [[buffer(2)]],
                    constant float* lamda [[buffer(3)]],
                    uint2 position [[thread_position_in_grid]],
                    uint2 threads [[threads_per_grid]]) {
    matrix.cols = 2;
    matrix.rows = 2;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 2 && position.y < 2) {
        if (position.y == 0 && position.x == 0) {
            matrix.data[index] = cos(*theta / 2.0f);
        }else if(position.y == 0 && position.x == 1) {
            matrix.data[index] = -complexExp(i * *lamda) * sin(*theta / 2.0f);
        }else if(position.y == 1 && position.y == 0) {
            matrix.data[index] = complexExp(i * *phi) * sin(*theta / 2.0f);
        }else {
            matrix.data[index] = complexExp(i * (*lamda + *phi)) * cos(*theta / 2.0f);
        }
    }
}

kernel void BARENCOMatrix(device ComplexMatrix& matrix [[buffer(0)]],
                    constant float* alpha [[buffer(1)]],
                    constant float* phi [[buffer(2)]],
                    constant float* theta [[buffer(3)]],
                    uint2 position [[thread_position_in_grid]],
                    uint2 threads [[threads_per_grid]]) {
    matrix.cols = 4;
    matrix.rows = 4;
    uint index = position.y * matrix.cols + position.x;
    
    if (position.x < 4 && position.y < 4) {
        if ((position.y == 2 && position.x == 2) ||
            (position.y == 3 && position.x == 3)) {
            matrix.data[index] = complexExp(i * *alpha) * cos(*theta);
        }else if(position.y == 3 && position.x == 2) {
            matrix.data[index] = -i * complexExp(i * (*alpha + *phi)) * sin(*theta);
        }else if(position.y == 2 && position.y == 3) {
            matrix.data[index] = -i * complexExp(i * (*alpha - *phi)) * sin(*theta);
        }else if((position.y == 0 && position.x == 0) ||
                 (position.y == 1 && position.x == 1)) {
            matrix.data[index] = ONE;
        }else {
            matrix.data[index] = ZERO;
        }
    }
}
