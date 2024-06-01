#ifndef GATE_H
#define GATE_H

#include <cuda_runtime.h>

__constant__ extern int I_GATE[2][2];
__constant__ extern int NOT_GATE[2][2];
__constant__ extern int Z_GATE[2][2];
__constant__ extern double HADAMARD_GATE[2][2];
__constant__ extern int CNOT_GATE[4][4];
__constant__ extern int CZ_GATE[4][4];
__constant__ extern int TOFFOLI_GATE[8][8];
__constant__ extern int SWAP_GATE[4][4];

__global__ void Ry(double theta, double* matrix);

#endif
