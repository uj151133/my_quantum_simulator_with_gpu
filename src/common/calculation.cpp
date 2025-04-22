#include "calculation.hpp"
#include "../models/uniqueTable.hpp"

size_t calculation::generateUniqueTableKey(const shared_ptr<QMDDNode>& node) {
    std::vector<uint8_t> buffer;

    for (const auto& edgeRow : node->edges) {
        for (const auto& edge : edgeRow) {
            double real = edge.weight.real();
            double imag = edge.weight.imag();
            const uint8_t* real_bytes = reinterpret_cast<const uint8_t*>(&real);
            const uint8_t* imag_bytes = reinterpret_cast<const uint8_t*>(&imag);

            buffer.insert(buffer.end(), real_bytes, real_bytes + sizeof(double));
            buffer.insert(buffer.end(), imag_bytes, imag_bytes + sizeof(double));

            const uint8_t* key_bytes = reinterpret_cast<const uint8_t*>(&edge.uniqueTableKey);
            buffer.insert(buffer.end(), key_bytes, key_bytes + sizeof(size_t));
        }
    }
    
    return static_cast<size_t>(XXH3_64bits(buffer.data(), buffer.size()));
}

size_t calculation::generateOperationCacheKey(const OperationKey& key) {
    auto customHash = [](const complex<double>& c) {
        size_t realHash = hash<double>()(c.real());
        size_t imagHash = hash<double>()(c.imag());
        return realHash ^ (imagHash << 1);
    };

    auto hash_combine = [](size_t& seed, size_t hash) {
        seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };

    auto edgeHash = [&](const QMDDEdge& edge) {
        size_t h = 0;
        hash_combine(h, customHash(edge.weight));
        hash_combine(h, hash<size_t>()(edge.uniqueTableKey));
        return h;
    };

    size_t seed = 0;

    if (get<1>(key) == OperationType::ADD) {
        size_t h1 = edgeHash(get<0>(key));
        size_t h2 = edgeHash(get<2>(key));
        if (h1 < h2) {
            hash_combine(seed, h1);
            hash_combine(seed, hash<OperationType>()(get<1>(key)));
            hash_combine(seed, h2);
        } else {
            hash_combine(seed, h2);
            hash_combine(seed, hash<OperationType>()(get<1>(key)));
            hash_combine(seed, h1);
        }
    } else {
        hash_combine(seed, edgeHash(get<0>(key)));
        hash_combine(seed, hash<OperationType>()(get<1>(key)));
        hash_combine(seed, edgeHash(get<2>(key)));
    }

    return seed;
}