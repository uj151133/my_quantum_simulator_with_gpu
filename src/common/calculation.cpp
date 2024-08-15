#include "calculation.hpp"
#include "../models/uniqueTable.hpp"

size_t calculation::generateUniqueTableKey(const QMDDNode& node, size_t row, size_t col, size_t rowStride, size_t colStride, const complex<double>& parentWeight) {
    auto customHash = [](const complex<double>& c) {
        size_t realHash = hash<double>()(c.real());
        size_t imagHash = hash<double>()(c.imag());
        return realHash ^ (imagHash << 1);
    };

    auto hashMatrixElement = [&](const complex<double>& value, size_t row, size_t col) {
        size_t valueHash = customHash(value);
        size_t elementHash = valueHash ^ (row + col + 0x9e3779b9 + (valueHash << 6) + (valueHash >> 2));
        return elementHash;
    };

    size_t hashValue = 0;
    UniqueTable& table = UniqueTable::getInstance();

    for (size_t i = 0; i < node.edges.size(); ++i) {
        size_t newRow = row + (i / 2) * rowStride;
        size_t newCol = col + (i % 2) * colStride;

        complex<double> combinedWeight = parentWeight * node.edges[i].weight;

        size_t elementHash;
        if (node.edges[i].isTerminal || node.edges[i].uniqueTableKey == 0) {
            elementHash = hashMatrixElement(combinedWeight, newRow, newCol);
        } else {
            shared_ptr<QMDDNode> foundNode = table.find(node.edges[i].uniqueTableKey);
            if (foundNode) {
                elementHash = generateUniqueTableKey(*foundNode, newRow, newCol, rowStride * 2, colStride * 2, combinedWeight);
            } else {
                elementHash = 0;
            }
        }

        hashValue ^= (elementHash + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2));
    }

    return hashValue;
}