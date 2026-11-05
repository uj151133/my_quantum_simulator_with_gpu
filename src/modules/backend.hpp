#ifndef BACKEND_HPP
#define BACKEND_HPP

#include "../models/qmdd.hpp"
#include "../models/arrayLeaf.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <unordered_map>

#if defined(USE_METAL)
using gpu_precision = float;
#else
using gpu_precision = double;
#endif

using namespace std;

struct GPUEdge {
    gpu_precision re = 0.0;
    gpu_precision im = 0.0;
    int32_t childIndex = -1;
};

struct GPUQMDD {
    vector<GPUEdge> edges;
};

struct GPUSV {
    void* reHandle = nullptr;
    void* imHandle = nullptr;
};

// 行列の意味上の形（ビットで持つ。両方立つこともある：スカラーや零行列）
constexpr uint32_t SHAPE_COL  = 1u;   // 第 0 列以外がゼロ
constexpr uint32_t SHAPE_DIAG = 2u;   // 非対角がゼロ

struct GPUInput {
    SonKind kind;

    gpu_precision root_re = 1.0;
    gpu_precision root_im = 0.0;

    GPUQMDD qmdd;
    GPUSV sv;

    uint32_t dim = 0;
    uint32_t pad = 1;

    uint32_t shape = 0;               // SHAPE_* の組（pad を掛ける前の形）
    uint32_t store = STORE_GENERAL;   // ArrayLeaf のバッファの持ち方（LeafStore）
};

inline uint32_t effDim(const GPUInput& in) { return in.dim * in.pad; }

enum class GpuOp { Mul, Add, Kron };

// 結果の密葉をどの持ち方で作るか。オペランドの形から決め、config（leaf.column / leaf.diagonal）で絞る。
uint32_t outputStore(GpuOp op, const GPUInput& A, const GPUInput& B);

GPUInput flattenQMDD(const QMDDEdge& root);

GPUInput wrapSV(const QMDDEdge& root);

GPUInput farewell(const QMDDEdge& root);

#endif