#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

__constant__ int I_GATE[2][2] = {
    {1, 0},
    {0, 1}
};

__constant__ int NOT_GATE[2][2] = {
    {0, 1},
    {1, 0}
};

__constant__ int Z_GATE[2][2] = {
    {1, 0},
    {0, -1}
};

__constant__ double HADAMARD_GATE[2][2] = {
    {1.0 / sqrt(2), 1.0 / sqrt(2)},
    {1.0 / sqrt(2), -1.0 / sqrt(2)}
};

__constant__ int CNOT_GATE[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1},
    {0, 0, 1, 0}
};

__constant__ int CZ_GATE[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, -1}
};

__constant__ int TOFFOLI_GATE[8][8] = {
    {1, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0, 1, 0}
};

__constant__ int SWAP_GATE[4][4] = {
    {1, 0, 0, 0},
    {0, 0, 1, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1}
};

__global__ void Ry(double theta, double* matrix) {
    matrix[0] = cos(theta / 2);
    matrix[1] = -sin(theta / 2);
    matrix[2] = sin(theta / 2);
    matrix[3] = cos(theta / 2);
}
