#ifndef UNIQUETABLE_HPP
#define UNIQUETABLE_HPP

#include "qmdd.hpp"
#include <unordered_map>
#include <functional>
#include <fstream>

using namespace std; 

// 再帰的なハッシュ関数の定義
struct QMDDNodeHash {
    size_t operator()(const QMDDNode& node) const;
    
    private:
    size_t customHash(const complex<double>& c) const;
    size_t hashMatrixElement(const complex<double>& value, size_t row, size_t col) const;
    void calculateMatrixHash(const QMDDNode& node, size_t row, size_t col, size_t rowStride, size_t colStride, size_t& hashValue) const;
};

// ユニークテーブルの定義
class UniqueTable {
private:
    unordered_map<QMDDNode, QMDDNode*, QMDDNodeHash> table;
    UniqueTable() = default;

public:
    static UniqueTable& getInstance() {
        static UniqueTable instance;
        return instance;
    }

    QMDDNode* find(size_t uniqueTableKey);
    size_t insert(QMDDNode* node);
    void saveToFile() const;

    // コピーコンストラクタと代入演算子を削除して、シングルトンであることを保証する
    UniqueTable(const UniqueTable&) = delete;
    UniqueTable& operator=(const UniqueTable&) = delete;
};

#endif // UNIQUETABLE_HPP
