#ifndef UNIQUETABLE_HPP
#define UNIQUETABLE_HPP

#include <atomic>
#include <vector>
#include <memory>
#include <iostream>
// #include <shared_mutex>
#include "../common/config.hpp"
#include "qmdd.hpp"

using namespace std;

// struct Entry {
//     long long key;
//     shared_ptr<QMDDNode> value;
//     Entry(long long k, shared_ptr<QMDDNode> v) : key(k), value(v) {}
// };

struct Entry {
    long long key;
    shared_ptr<QMDDNode> value;
    Entry* next;
    Entry(long long k, shared_ptr<QMDDNode> v, Entry* n=nullptr) : key(k), value(v), next(n) {}
};

class UniqueTable {
private:
    vector<atomic<Entry*>> table;
    static constexpr long long tableSize=1048576;
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
