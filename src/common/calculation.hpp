#ifndef CALCULATION_HPP
#define CALCULATION_HPP


#include <string>
#include <utility>
#include <typeinfo>
#include <xxhash.h>
#include "../models/qmdd.hpp"

using namespace std;

namespace calculation {
    size_t generateUniqueTableKey(const shared_ptr<QMDDNode>& node);
    size_t generateOperationCacheKey(const OperationKey& key);
}
#endif
