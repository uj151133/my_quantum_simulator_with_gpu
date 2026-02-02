#include "calculation.hpp"

int64_t calculation::generateUniqueTableKey(const shared_ptr<QMDDNode>& node) {
    vector<uint8_t> buffer;
    size_t rowIdx = 0;
    for (const auto& edgeRow : node->edges) {
        size_t colIdx = 0;
        for (const auto& edge : edgeRow) {
            const uint8_t* rowIdx_bytes = reinterpret_cast<const uint8_t*>(&rowIdx);
            buffer.insert(buffer.end(), rowIdx_bytes, rowIdx_bytes + sizeof(size_t));
            const uint8_t* colIdx_bytes = reinterpret_cast<const uint8_t*>(&colIdx);
            buffer.insert(buffer.end(), colIdx_bytes, colIdx_bytes + sizeof(size_t));
            double real = edge.weight.real();
            double imag = edge.weight.imag();
            const uint8_t* real_bytes = reinterpret_cast<const uint8_t*>(&real);
            const uint8_t* imag_bytes = reinterpret_cast<const uint8_t*>(&imag);

            buffer.insert(buffer.end(), real_bytes, real_bytes + sizeof(double));
            buffer.insert(buffer.end(), imag_bytes, imag_bytes + sizeof(double));

            const uint8_t* key_bytes = reinterpret_cast<const uint8_t*>(&edge.uniqueTableKey);
            buffer.insert(buffer.end(), key_bytes, key_bytes + sizeof(size_t));
            colIdx++;
        }
        rowIdx++;
    }
    // return llabs(XXH3_64bits(buffer.data(), buffer.size()));

    return XXH3_64bits(buffer.data(), buffer.size());
}

int64_t calculation::generateOperationCacheKey(const OperationKey& key) {
    const int64_t data[2] = { key.first, key.second };
    const uint64_t h = XXH3_64bits(data, sizeof(data));

    // int64_t で返したいなら符号ビットを落としておく（負値回避）
    return static_cast<int64_t>(h & 0x7fffffffffffffffULL);
}