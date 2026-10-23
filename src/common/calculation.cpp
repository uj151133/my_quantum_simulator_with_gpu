#include "calculation.hpp"
#include <cstring>

int64_t calculation::generateUniqueTableKey(const QMDDNode* node) {
    uint64_t buf[12];
    size_t n = 0;
    for (const auto& edgeRow : node->edges) {
        for (const auto& edge : edgeRow) {
            const double re = edge.weight.real();
            const double im = edge.weight.imag();
            uint64_t reBits, imBits;
            memcpy(&reBits, &re, sizeof(reBits));
            memcpy(&imBits, &im, sizeof(imBits));
            buf[n++] = static_cast<uint64_t>(edge.key_);
            buf[n++] = reBits;
            buf[n++] = imBits;
        }
    }
    const uint64_t h = XXH3_64bits(buf, sizeof(buf)) & 0x7FFFFFFFFFFFFFFFULL;
    return static_cast<int64_t>(h == 0 ? 1 : h);
}

int64_t calculation::generateOperationCacheKey(const OperationKey& key) {
    const int64_t data[2] = { key.first, key.second };
    const uint64_t h = XXH3_64bits(data, sizeof(data));

    // int64_t で返したいなら符号ビットを落としておく（負値回避）
    return static_cast<int64_t>(h & 0x7fffffffffffffffULL);
}