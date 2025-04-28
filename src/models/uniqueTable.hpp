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

struct Entry {
    size_t key;
    shared_ptr<QMDDNode> value;
    Entry* next;
    Entry(size_t k, shared_ptr<QMDDNode> v, Entry* n=nullptr) : key(k), value(v), next(n) {}
};

class UniqueTable {
private:
    // unordered_map<size_t, vector<Entry>> table;
    vector<atomic<Entry*>> table;
    // mutable shared_mutex tableMutex;
    static constexpr size_t tableSize=1048576;
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
