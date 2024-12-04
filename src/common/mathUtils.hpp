#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include <cmath>
#include <numeric>
#include <functional>
#include <thread>
#include "../models/qmdd.hpp"
#include "../models/uniqueTable.hpp"
#include "../models/operationCache.hpp"
#include "calculation.hpp"

using namespace std;

namespace mathUtils {
    QMDDEdge mul(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge mulParallel(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge mulForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1);

    QMDDEdge add(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge addParallel(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge addForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1);

    QMDDEdge kron(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge kronParallel(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge kronForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1);


    double csc(double theta);
    complex<double> csc(complex<double> theta);
    double sec(double theta);
    complex<double> sec(complex<double> theta);
    double cot(double theta);
    complex<double> cot(complex<double> theta);

    double sumOfSquares(const vector<complex<double>>& vec);
}

#endif // MATH_UTILS_HPP
