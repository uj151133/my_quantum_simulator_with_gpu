#ifndef MEMO_HPP
#define MEMO_HPP

#include <atomic>
#include <vector>
#include <memory>
#include <iostream>
#include <boost/fiber/all.hpp>
#include "../common/parameter.hpp"
#include "sv.hpp"

using namespace std;

struct SVEntry {
    int64_t key;
    weak_ptr<SVLeaf> value;
    SVEntry* next;
    SVEntry(int64_t k, weak_ptr<SVLeaf> v, SVEntry* n=nullptr) : key(k), value(v), next(n) {}
};

class Memo {
private:
    const int64_t tableSize_;
    vector<atomic<SVEntry*>> table_;
    Memo();
    int64_t hash(int64_t hashKey) const;

public:
    Memo(const Memo&) = delete;
    Memo& operator=(const Memo&) = delete;
    static Memo& getInstance();

    void insert(int64_t hashKey, const shared_ptr<SVLeaf>& leaf);
    shared_ptr<SVLeaf> find(int64_t hashKey) const;

    void printAllEntries() const;
    int getTotalEntryCount() const;
};

#endif