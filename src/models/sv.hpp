#ifndef SV_HPP
#define SV_HPP

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif
    void releaseGpuBuffer(void* p);
#ifdef __cplusplus
}
#endif

struct SVLeaf {
    size_t  dim;
    void*   reBuf;
    void*   imBuf;

    SVLeaf(size_t dim, void* reBuf, void* imBuf);
    ~SVLeaf();
};


#endif