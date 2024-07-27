#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include "../models/qmdd.hpp"
#include <complex>
#include <vector>

using namespace std;

namespace mathUtils {
    QMDDEdge mul(const QMDDEdge& m, const QMDDEdge& v);
    QMDDEdge add(const QMDDEdge& e1, const QMDDEdge& e2);
}

#endif // MATH_UTILS_HPP
