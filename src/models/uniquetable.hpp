#ifndef UNIQUETABLE_HPP
#define UNIQUETABLE_HPP

#include <unordered_map>
#include <vector>
#include <memory>
#include "qmdd.hpp"

class UniqueTable {
private:
    std::unordered_map<size_t, std::vector<std::shared_ptr<QMDDNode>>> table;

    UniqueTable() = default;

public:
    UniqueTable(const UniqueTable&) = delete;
    UniqueTable& operator=(const UniqueTable&) = delete;

    static UniqueTable& getInstance();

    void insertNode(size_t hashKey, std::shared_ptr<QMDDNode> node);
    std::shared_ptr<QMDDNode> findNode(size_t hashKey, std::shared_ptr<QMDDNode> node);
};

#endif // UNIQUETABLE_H
