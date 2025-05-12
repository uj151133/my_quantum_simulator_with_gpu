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
    long long generateUniqueTableKey(const shared_ptr<QMDDNode>& node);
    long long generateOperationCacheKey(const OperationKey& key);
}
#endif
