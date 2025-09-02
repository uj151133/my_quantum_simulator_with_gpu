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
    int64_t key;
    shared_ptr<QMDDNode> value;
    Entry* next;
    Entry(int64_t k, shared_ptr<QMDDNode> v, Entry* n=nullptr) : key(k), value(v), next(n) {}
};

class UniqueTable {
private:
    vector<atomic<Entry*>> table;
    static constexpr int64_t tableSize=1048576;
    UniqueTable();
    int64_t hash(int64_t hashKey) const;



public:
    UniqueTable(const UniqueTable&) = delete;
    UniqueTable& operator=(const UniqueTable&) = delete;
    static UniqueTable& getInstance();
    void insert(int64_t hashKey, shared_ptr<QMDDNode> node);
    shared_ptr<QMDDNode> find(int64_t hashKey) const;
    void printAllEntries() const;
    void printNodeNum() const;
    int getTotalEntryCount() const;
};

#endif
