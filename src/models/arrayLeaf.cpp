#include "arrayLeaf.hpp"

ArrayLeaf::ArrayLeaf(size_t dim, void* reBuf, void* imBuf, int64_t id, uint32_t store)
    : reBuf(reBuf), imBuf(imBuf), id(id),
      level(static_cast<int16_t>(dim > 1 ? 63 - __builtin_clzll(dim) : 0)),
      store(static_cast<uint16_t>(store)) {}

ArrayLeaf::~ArrayLeaf() {
        releaseGpuBuffer(this->reBuf);
        releaseGpuBuffer(this->imBuf);
}

ostream& operator<<(std::ostream& os, const ArrayLeaf& sv) {
    os << "ArrayLeaf { "
        << "dim = " << sv.dim()
        << ", store = " << sv.store
        << ", reBuf = " << sv.reBuf
        << ", imBuf = " << sv.imBuf
        << " }";

    return os;
}