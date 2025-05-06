#ifndef UNIQUETABLE_HPP
#define UNIQUETABLE_HPP

#include <mutex>
#include <shared_mutex>
#include "../common/config.hpp"
#include "qmdd.hpp"

using namespace std;

struct Entry {
    long long key;
    shared_ptr<QMDDNode> value;
    Entry(long long k, shared_ptr<QMDDNode> v) : key(k), value(v) {}
};

class UniqueTable {
private:
    unordered_map<long long, vector<Entry>> table;
    mutable shared_mutex tableMutex;
    const size_t tableSize ;
    UniqueTable();
    long long hash(long long hashKey) const;



public:
    UniqueTable(const UniqueTable&) = delete;
    UniqueTable& operator=(const UniqueTable&) = delete;
    static UniqueTable& getInstance();
    void insert(long long hashKey, shared_ptr<QMDDNode> node);
    shared_ptr<QMDDNode> find(long long hashKey) const;
    void printAllEntries() const;
};

#endif
