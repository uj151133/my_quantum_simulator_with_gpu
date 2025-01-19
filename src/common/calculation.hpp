#ifndef CALCULATION_HPP
#define CALCULATION_HPP


#include <string>
#include <utility>
#include <typeinfo>
#include "../models/qmdd.hpp"

using namespace std;

namespace calculation {
    size_t generateUniqueTableKey(const shared_ptr<QMDDNode>& node, size_t row = 0, size_t col = 0, size_t rowStride = 1, size_t colStride = 1, const complex<double>& parentWeight = complex<double>(1.0, 0.0));
    size_t generateOperationCacheKey(const OperationKey& key);
}
#endif
