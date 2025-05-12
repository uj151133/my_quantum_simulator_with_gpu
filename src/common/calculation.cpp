#include "calculation.hpp"
#include "../models/uniqueTable.hpp"

long long calculation::generateUniqueTableKey(const shared_ptr<QMDDNode>& node) {
    vector<uint8_t> buffer;

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

    return llabs(XXH3_64bits(buffer.data(), buffer.size()));
}

long long calculation::generateOperationCacheKey(const OperationKey& key) {
    auto customHash = [](const complex<double>& c) {
        size_t realHash = std::hash<double>()(c.real());
        size_t imagHash = std::hash<double>()(c.imag());
        return realHash ^ (imagHash << 1);
    };

    auto hash_combine = [](long long& seed, size_t hash) {
        seed ^= static_cast<long long>(hash) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };

    auto edgeHash = [&](const QMDDEdge& edge) {
        long long h = 0;
        hash_combine(h, customHash(edge.weight));
        hash_combine(h, std::hash<size_t>()(edge.uniqueTableKey));
        return h;
    };

    long long seed = 0;

    if (std::get<1>(key) == OperationType::ADD) {
        long long h1 = edgeHash(std::get<0>(key));
        long long h2 = edgeHash(std::get<2>(key));
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