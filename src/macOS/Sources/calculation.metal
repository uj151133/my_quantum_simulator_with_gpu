#include <metal_stdlib>
using namespace metal;

kernel void hashMatrixElement(
    device const float2* values [[ buffer(0) ]],
    device const int* rows [[ buffer(1) ]],
    device const int* cols [[ buffer(2) ]],
    device uint* results [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]) {

    float2 value = values[id];
    int row = rows[id];
    int col = cols[id];

    uint valueHash = hash(value.x) ^ (hash(value.y) << 1);
    uint rowHash = uint(row) << 16;
    uint colHash = uint(col) & 0xFFFF;
    uint combinedHash = rowHash | colHash;
    uint elementHash = valueHash ^ combinedHash ^ 0x9e3779b9 + (valueHash << 6) + (valueHash >> 2);

    results[id] = elementHash;
}

uint hash(float x) {
    return uint(bitcast<uint>(x));
}