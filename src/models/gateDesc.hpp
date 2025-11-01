#ifndef GATEDESC_HPP
#define GATEDESC_HPP
#include <string>
#include <vector>
#include <cstdint>

using namespace std;

struct GateDesc {
    string tag;           // "H","RZ","CX","CZ","CRZ","FUSED",...
    vector<int> qubits;   // 作用量子ビット（FUSEDは集合）
    bool is_diag = false;      // 対角なら true
    bool is_fused = false;     // FUSEDなら true
    uint64_t handle = 0;       // FUSED 実体のストアID（非FUSEDは0）
    size_t edge_nodes = 0;     // QMDDEdgeのノード数（近似可）
};