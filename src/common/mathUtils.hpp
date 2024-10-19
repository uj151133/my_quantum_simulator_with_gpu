#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include <boost/fiber/all.hpp>
#include <cmath>
#include "../models/qmdd.hpp"
#include "../models/uniqueTable.hpp"
#include "../models/operationCache.hpp"
#include "calculation.hpp"

using namespace std;

namespace mathUtils {
    QMDDEdge mul(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge add(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge kron(const QMDDEdge& e0, const QMDDEdge& e1);

    complex<double> csc(complex<double> theta);
    complex<double> sec(complex<double> theta);
    complex<double> cot(complex<double> theta);
}

#endif // MATH_UTILS_HPP
