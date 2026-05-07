#ifndef BACKEND_HPP
#define BACKEND_HPP

#include "../models/qmdd.hpp"
#include "../models/sv.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(USE_METAL)
using gpu_precision = float;
#else
using gpu_precision = double;
#endif

struct GPUEdge {
    gpu_precision re = 0.0;
    gpu_precision im = 0.0;
    int32_t childIndex = -1;
};

struct GPUQMDD {
    std::vector<GPUEdge> edges;
};

struct GPUSV {
    void* reHandle = nullptr;
    void* imHandle = nullptr;
};

struct GPUInput {
    enum class Kind : uint8_t { QMDD, SV, Terminal };
    Kind kind;

    gpu_precision root_re = 1.0;
    gpu_precision root_im = 0.0;

    GPUQMDD qmdd;
    GPUSV sv;

    uint32_t dim = 0;
};

GPUInput flattenQMDD(const QMDDEdge& root);

GPUInput wrapSVEdge(const QMDDEdge& root);

#endif