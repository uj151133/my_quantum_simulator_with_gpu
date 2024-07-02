#include "uniquetable.hpp"
#include <iostream>

using namespace std; 

// QMDDNodeHashのoperator()の定義
size_t QMDDNodeHash::operator()(const QMDDNode& node) const {
    size_t hashValue = 0;
    for (const auto& edge : node.edges) {
        size_t edgeHash = customHash(edge.weight);
        if (!edge.isTerminal) {
            edgeHash ^= (*this)(*edge.node); // 再帰的にハッシュを計算
        }
        hashValue ^= edgeHash + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
    }
    return hashValue;
}

size_t QMDDNodeHash::customHash(const complex<double>& c) const {
    return hash<double>()(c.real()) ^ hash<double>()(c.imag());
}

// UniqueTableのfindメソッドの定義
QMDDNode* UniqueTable::find(const QMDDNode& node) {
    auto it = table.find(node);
    if (it != table.end()) {
        return it->second;
    }
    return nullptr;
}

// UniqueTableのinsertメソッドの定義
size_t UniqueTable::insert(QMDDNode* node) {
    table[*node] = node;
    return table.size(); // キーとしてサイズを返す
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
