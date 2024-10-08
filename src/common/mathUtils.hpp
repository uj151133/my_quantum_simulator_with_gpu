#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include "../models/qmdd.hpp"
#include "../models/uniqueTable.hpp"
#include "../models/operationCache.hpp"
#include "calculation.hpp"

using namespace std;

namespace mathUtils {
    QMDDEdge multiplication(const QMDDEdge& m, const QMDDEdge& v);
    QMDDEdge mul(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge addition(const QMDDEdge& e1, const QMDDEdge& e2);
    QMDDEdge add(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge kroneckerProduct(const QMDDEdge& e1, const QMDDEdge& e2);
    QMDDEdge kron(const QMDDEdge& e0, const QMDDEdge& e1);

    QMDDEdge mulAny(QMDDEdge& e, int times);
}

#endif // MATH_UTILS_HPP
