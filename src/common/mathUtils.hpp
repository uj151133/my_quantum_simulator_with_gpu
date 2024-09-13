#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include <boost/fiber/all.hpp>

#include "../models/qmdd.hpp"
#include "../models/uniqueTable.hpp"
#include "../models/operationCache.hpp"
#include "calculation.hpp"

using namespace std;

namespace mathUtils {
    QMDDEdge multiplication(const QMDDEdge& m, const QMDDEdge& v);
    QMDDEdge addition(const QMDDEdge& e1, const QMDDEdge& e2);
    QMDDEdge kroneckerProduct(const QMDDEdge& e1, const QMDDEdge& e2);
}

#endif // MATH_UTILS_HPP
