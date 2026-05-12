#ifndef SV_HPP
#define SV_HPP

#include <cstdint>
#include <cstddef>
#include <ostream>

#ifdef __cplusplus
extern "C" {
#endif
    void releaseGpuBuffer(void* p);
#ifdef __cplusplus
}
#endif

extern "C" bool copyGpuBufferToHostFloat(void* gpuHandle, float* dst, size_t count);

using namespace std;

struct SVLeaf {
    size_t  dim;
    void*   reBuf;
    void*   imBuf;

    SVLeaf(size_t dim, void* reBuf, void* imBuf);
    ~SVLeaf();
};
ostream& operator<<(std::ostream& os, const SVLeaf& sv);


#endif