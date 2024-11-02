#ifndef UNIQUETABLE_HPP
#define UNIQUETABLE_HPP

#include <shared_mutex>
#include "qmdd.hpp"

using namespace std;

class UniqueTable {
private:
    unordered_map<size_t, vector<shared_ptr<QMDDNode>>> table;
    mutable shared_mutex tableMutex;
    UniqueTable() = default;

public:
    UniqueTable(const UniqueTable&) = delete;
    UniqueTable& operator=(const UniqueTable&) = delete;
    static UniqueTable& getInstance();
    void insert(size_t hashKey, shared_ptr<QMDDNode> node);
    shared_ptr<QMDDNode> find(size_t hashKey) const;
    void printAllEntries() const;
};

#endif
