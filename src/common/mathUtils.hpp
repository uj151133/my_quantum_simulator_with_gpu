#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include "../models/qmdd.hpp"
#include "../models/uniqueTable.hpp"
#include <complex>
#include <vector>

using namespace std;

namespace mathUtils {
    QMDDEdge multiplication(const QMDDEdge& m, const QMDDEdge& v);
    QMDDEdge addition(const QMDDEdge& e1, const QMDDEdge& e2);
}

#endif // MATH_UTILS_HPP
