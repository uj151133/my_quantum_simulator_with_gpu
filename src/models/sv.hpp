#ifndef SV_HPP
#define SV_HPP

#include <cstdint>
#include <cstddef>

struct SVLeaf {
    int64_t sourceKey;   // 変換元サブツリーの uniqueTableKey
    size_t  size;    // 要素数（2^k）
    void*   reBuf;  // GPU側メモリ（実部）Metal: id<MTLBuffer>、CUDA: double*
    void*   imBuf;  // GPU側メモリ（虚部）
    bool    valid = false;
};

#endif