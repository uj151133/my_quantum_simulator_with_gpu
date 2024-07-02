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
};

// ユニークテーブルの定義
class UniqueTable {
private:
    unordered_map<QMDDNode, QMDDNode*, QMDDNodeHash> table;

public:
    QMDDNode* find(const QMDDNode& node);
    size_t insert(QMDDNode* node);
    void saveToFile() const;
};

#endif // UNIQUETABLE_HPP
