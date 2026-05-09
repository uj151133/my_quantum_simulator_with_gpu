#include "sv.hpp"

SVLeaf::SVLeaf(size_t dim, void* reBuf, void* imBuf) : dim(dim), reBuf(reBuf), imBuf(imBuf) {}

SVLeaf::~SVLeaf() {
        releaseGpuBuffer(reBuf);
        releaseGpuBuffer(imBuf);
}