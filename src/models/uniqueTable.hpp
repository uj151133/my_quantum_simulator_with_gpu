#ifndef UNIQUETABLE_HPP
#define UNIQUETABLE_HPP

#include <mutex>
#include <shared_mutex>
#include "../common/config.hpp"
#include "qmdd.hpp"

using namespace std;

struct Entry {
    size_t key;
    weak_ptr<QMDDNode> value;
    Entry(size_t k, weak_ptr<QMDDNode> v) : key(k), value(v) {}

};

class UniqueTable {
private:
    unordered_map<size_t, vector<Entry>> table;
    mutable shared_mutex tableMutex;
    size_t tableSize ;
    UniqueTable();
    size_t hash(size_t hashKey) const;


public:
    UniqueTable(const UniqueTable&) = delete;
    UniqueTable& operator=(const UniqueTable&) = delete;
    static UniqueTable& getInstance();
    void insert(size_t hashKey, shared_ptr<QMDDNode> node);
    shared_ptr<QMDDNode> find(size_t hashKey) const;
    void printAllEntries() const;
};

#endif
