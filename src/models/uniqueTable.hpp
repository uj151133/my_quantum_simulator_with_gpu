#ifndef UNIQUETABLE_HPP
#define UNIQUETABLE_HPP

#include <mutex>
#include <shared_mutex>
#include "qmdd.hpp"

using namespace std;

struct Entry {
    size_t key;
    shared_ptr<QMDDNode> value;
    Entry(size_t k, shared_ptr<QMDDNode> v) : key(k), value(v) {}
};

class UniqueTable {
private:
    unordered_map<size_t, vector<Entry>> table;
    mutable shared_mutex tableMutex;
    UniqueTable() {
        table.reserve(ENTRY_COUNT);
    }
    size_t hash(size_t key) const;

public:
    UniqueTable(const UniqueTable&) = delete;
    UniqueTable& operator=(const UniqueTable&) = delete;
    static UniqueTable& getInstance();
    void insert(size_t hashKey, shared_ptr<QMDDNode> node);
    shared_ptr<QMDDNode> find(size_t hashKey) const;
    void printAllEntries() const;
};

#endif
