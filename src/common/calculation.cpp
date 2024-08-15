#include "calculation.hpp"
#include "../models/uniqueTable.hpp"

size_t calculation::customHash(const complex<double>& c) {
    size_t realHash = hash<double>()(c.real());
    size_t imagHash = hash<double>()(c.imag());
    // cout << "customHash: real(" << c.real() << ") => " << realHash << ", imag(" << c.imag() << ") => " << imagHash << endl;
    return realHash ^ (imagHash << 1);
}

size_t calculation::calculateMatrixHash(const QMDDNode& node, size_t row, size_t col, size_t rowStride, size_t colStride, const complex<double>& parentWeight) {
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
            // find() の結果をデリファレンスして calculateMatrixHash に渡す
            shared_ptr<QMDDNode> foundNode = table.find(node.edges[i].uniqueTableKey);
            if (foundNode) {
                elementHash = calculateMatrixHash(*foundNode, newRow, newCol, rowStride * 2, colStride * 2, combinedWeight);
            } else {
                elementHash = 0;
            }
        }

        hashValue ^= (elementHash + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2));
    }

    return hashValue;
}

size_t calculation::calculateMatrixHash(const QMDDNode& node) {
    return calculateMatrixHash(node, 0, 0, 1, 1, complex<double>(1.0, 0.0));
}

size_t calculation::hashMatrixElement(const complex<double>& value, size_t row, size_t col) {
    size_t valueHash = customHash(value);
    size_t elementHash = valueHash ^ (row + col + 0x9e3779b9 + (valueHash << 6) + (valueHash >> 2));
    // cout << "hashMatrixElement: value(" << value << "), row(" << row << "), col(" << col << ") => " << elementHash << endl;
    return elementHash;
}