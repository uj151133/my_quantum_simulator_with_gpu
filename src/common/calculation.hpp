#ifndef CALCULATION_HPP
#define CALCULATION_HPP

#include "../models/qmdd.hpp"
#include <complex>

using namespace std;


namespace calculation {
    size_t calculateMatrixHash(const QMDDNode& node);
    size_t calculateMatrixHash(const QMDDNode& node, size_t row, size_t col, size_t rowStride, size_t colStride, const complex<double>& parentWeight);
    size_t hashMatrixElement(const complex<double>& value, size_t row, size_t col);
    size_t customHash(const complex<double>& value);
}


#endif
