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

struct ArrayLeaf {
    void*   reBuf;
    void*   imBuf;
    int64_t id;
    int     level;
    std::atomic<uint32_t> refCount{0};
    ArrayLeaf(size_t dim, void* reBuf, void* imBuf, int64_t id = 0);

    size_t dim() const { return size_t(1) << this->level; }

    ArrayLeaf(const ArrayLeaf&) = delete;
    ArrayLeaf& operator=(const ArrayLeaf&) = delete;
    ~ArrayLeaf();
};
ostream& operator<<(std::ostream& os, const ArrayLeaf& sv);


#endif