#include "arrayLeaf.hpp"

ArrayLeaf::ArrayLeaf(size_t dim, void* reBuf, void* imBuf, int64_t id)
    : reBuf(reBuf), imBuf(imBuf), id(id), level(dim > 1 ? 63 - __builtin_clzll(dim) : 0) {}

ArrayLeaf::~ArrayLeaf() {
        releaseGpuBuffer(this->reBuf);
        releaseGpuBuffer(this->imBuf);
}

ostream& operator<<(std::ostream& os, const ArrayLeaf& sv) {
    os << "ArrayLeaf { "
        << "dim = " << sv.dim()
        << ", reBuf = " << sv.reBuf
        << ", imBuf = " << sv.imBuf
        << " }";

    return os;
}