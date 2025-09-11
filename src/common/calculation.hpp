#ifndef CALCULATION_HPP
#define CALCULATION_HPP


#include <string>
#include <utility>
#include <typeinfo>
#include <xxhash.h>
#include <cstdlib>
#include "../models/qmdd.hpp"

using namespace std;

namespace calculation {
    uint64_t generateUniqueTableKey(const shared_ptr<QMDDNode>& node);
    int64_t generateOperationCacheKey(const OperationKey& key);
}
#endif
