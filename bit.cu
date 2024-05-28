#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

__constant__ int  KET_0[1][2] = {
    {1},
    {0}
};

__constant__ int  KET_1[1][2] = {
    {0},
    {1}
};

__constant__ int  BRA_0[2] = {1, 0};

__constant__ int  BRA_1[2] = {0, 1};