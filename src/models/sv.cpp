#include "sv.hpp"
#include <atomic>

// プロファイル用：現在生きている SVLeaf の数（ctorで++/dtorで--）と、これまでの最大 dim
static std::atomic<int64_t> g_liveSVLeaf{0};
static std::atomic<size_t>  g_maxSVDim{0};
extern "C" int64_t liveSVLeafCount() { return g_liveSVLeaf.load(); }
extern "C" size_t  maxSVLeafDim()    { return g_maxSVDim.load(); }

SVLeaf::SVLeaf(size_t dim, void* reBuf, void* imBuf) : dim(dim), reBuf(reBuf), imBuf(imBuf) {
    g_liveSVLeaf.fetch_add(1, std::memory_order_relaxed);
    size_t cur = g_maxSVDim.load(std::memory_order_relaxed);
    while (dim > cur && !g_maxSVDim.compare_exchange_weak(cur, dim, std::memory_order_relaxed)) {}
}

SVLeaf::~SVLeaf() {
        g_liveSVLeaf.fetch_sub(1, std::memory_order_relaxed);
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