#ifndef ARRAYLEAF_HPP
#define ARRAYLEAF_HPP

#include <cstdint>
#include <cstddef>
#include <ostream>
#include <atomic>

#ifdef __cplusplus
extern "C" {
#endif
    void releaseGpuBuffer(void* p);
#ifdef __cplusplus
}
#endif

extern "C" bool copyGpuBufferToHostDouble(void* gpuHandle, double* dst, size_t count);

extern "C" bool copyGpuBufferToHostFloat(void* gpuHandle, float* dst, size_t count);

using namespace std;

// 密葉のバッファの持ち方。GPU カーネル（Heian.cu / Heian.metal）も同じ値を使う。
//   GENERAL  … dim×dim を行優先で全部持つ
//   COLUMN   … 第 0 列だけを持つ（dim 要素）。他の列はゼロ
//   DIAGONAL … 対角だけを持つ（dim 要素）。非対角はゼロ
enum LeafStore : uint32_t { STORE_GENERAL = 0, STORE_COLUMN = 1, STORE_DIAGONAL = 2 };

inline size_t leafElemCount(uint32_t store, size_t dim) {
    return store == STORE_GENERAL ? dim * dim : dim;
}

struct ArrayLeaf {
    void*   reBuf;
    void*   imBuf;
    int64_t id;
    int16_t level;
    uint16_t store;                       // LeafStore
    std::atomic<uint32_t> refCount{0};
    ArrayLeaf(size_t dim, void* reBuf, void* imBuf, int64_t id = 0, uint32_t store = STORE_GENERAL);

    size_t dim() const { return size_t(1) << this->level; }
    // バッファに実際に入っている要素数
    size_t elemCount() const { return leafElemCount(this->store, this->dim()); }

    ArrayLeaf(const ArrayLeaf&) = delete;
    ArrayLeaf& operator=(const ArrayLeaf&) = delete;
    ~ArrayLeaf();
};
ostream& operator<<(std::ostream& os, const ArrayLeaf& sv);


#endif