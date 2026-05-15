#include "sv.hpp"

SVLeaf::SVLeaf(size_t dim, void* reBuf, void* imBuf) : dim(dim), reBuf(reBuf), imBuf(imBuf) {}

SVLeaf::~SVLeaf() {
        releaseGpuBuffer(reBuf);
        releaseGpuBuffer(imBuf);
}

ostream& operator<<(std::ostream& os, const SVLeaf& sv) {
    os << "SVLeaf { "
        << "dim = " << sv.dim
        << ", reBuf = " << sv.reBuf
        << ", imBuf = " << sv.imBuf
        << " }";

    return os;
}