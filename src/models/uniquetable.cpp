#include "uniqueTable.hpp"
#include <iostream>

using namespace std; 

// QMDDNodeHashのoperator()の定義
size_t QMDDNodeHash::operator()(const QMDDNode& node) const {
    size_t hashValue = 0;
    calculateMatrixHash(node, 0, 0, 1, 1, hashValue);
    return hashValue;
}

size_t QMDDNodeHash::customHash(const complex<double>& c) const {
    return hash<double>()(c.real()) ^ hash<double>()(c.imag());
}

size_t QMDDNodeHash::hashMatrixElement(const complex<double>& value, size_t row, size_t col) const {
    size_t valueHash = customHash(value);
    return valueHash ^ (row + col + 0x9e3779b9 + (valueHash << 6) + (valueHash >> 2));
}

void QMDDNodeHash::calculateMatrixHash(const QMDDNode& node, size_t row, size_t col, size_t rowStride, size_t colStride, size_t& hashValue) const {
    for (size_t i = 0; i < node.edges.size(); ++i) {
        if (node.edges[i].isTerminal || !node.edges[i].node) {
            size_t newRow = row + (i / 2) * rowStride;
            size_t newCol = col + (i % 2) * colStride;
            hashValue ^= hashMatrixElement(node.edges[i].weight, newRow, newCol);
        } else {
            calculateMatrixHash(*node.edges[i].node, row, col, rowStride * 2, colStride * 2, hashValue);
        }
    }
}

// UniqueTableのfindメソッドの定義
QMDDNode* UniqueTable::find(size_t uniqueTableKey) {
    auto it = table.find(uniqueTableKey);
    if (it != table.end()) {
        // ノードが見つかった場合、そのポインタを返す
        return it->second;
    }

    // ノードが見つからなかった場合、nullptrを返す
    return nullptr;
}


// UniqueTableのinsertメソッドの定義
size_t UniqueTable::insert(QMDDNode* node) {
    if (!node) {
        throw std::invalid_argument("Attempted to insert a null node into the UniqueTable.");
    }

    QMDDNodeHash hashFunc;
    size_t hashValue = hashFunc(*node);

    auto it = table.find(node->uniqueTableKey);
    if (it != table.end()) {
        // 既存のノードがテーブルにある場合、そのノードの uniqueTableKey を返す
        return it->second->uniqueTableKey;
    }

    // ノードがテーブルに存在しない場合、新しいノードを追加し、uniqueTableKey にハッシュ値を設定
    node->uniqueTableKey = hashValue;
    table[node->uniqueTableKey] = node;
    return node->uniqueTableKey;
}


// UniqueTableのsaveToFileメソッドの定義
void UniqueTable::saveToFile() const {
    ofstream outfile("unique_table.txt");

    if (!outfile.is_open()) {
        cerr << "ファイルを開くことができませんでした: " << "unique_table.txt" << endl;
        return;
    }

    for (const auto& pair : table) {
        outfile << "Node KEY: " << pair.first.uniqueTableKey << ", Node VALUE: " << pair.second << endl;
    }

    outfile.close();
}
