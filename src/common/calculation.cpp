#include "calculation.hpp"
#include "../models/uniqueTable.hpp"

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
    return llabs(XXH3_64bits(buffer.data(), buffer.size()));
}

int64_t calculation::generateOperationCacheKey(const OperationKey& key) {
    auto customHash = [](const complex<double>& c) {
        size_t realHash = std::hash<double>()(c.real());
        size_t imagHash = std::hash<double>()(c.imag());
        return realHash ^ (imagHash << 1);
    };

    auto hash_combine = [](int64_t& seed, size_t hash) {
        seed ^= static_cast<int64_t>(hash) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };

    auto edgeHash = [&](const QMDDEdge& edge) {
        int64_t h = 0;
        hash_combine(h, customHash(edge.weight));
        hash_combine(h, std::hash<size_t>()(edge.uniqueTableKey));
        return h;
    };

    int64_t seed = 0;

    if (std::get<1>(key) == OperationType::ADD) {
        int64_t h1 = edgeHash(std::get<0>(key));
        int64_t h2 = edgeHash(std::get<2>(key));
        if (h1 < h2) {
            hash_combine(seed, h1);
            hash_combine(seed, std::hash<OperationType>()(std::get<1>(key)));
            hash_combine(seed, h2);
        } else {
            hash_combine(seed, h2);
            hash_combine(seed, std::hash<OperationType>()(std::get<1>(key)));
            hash_combine(seed, h1);
        }
    } else {
        hash_combine(seed, edgeHash(std::get<0>(key)));
        hash_combine(seed, std::hash<OperationType>()(std::get<1>(key)));
        hash_combine(seed, edgeHash(std::get<2>(key)));
    }

    return llabs(seed);
}