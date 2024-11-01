#include "constant.hpp"
#include <iostream>

complex<double> i(0.0, 1.0);
QMDDEdge edgeZero = QMDDEdge(.0, nullptr);
QMDDEdge edgeOne = QMDDEdge(1.0, nullptr);
once_flag initFlag;

void init() {
    // cout << "Initializing constant values" << std::endl;
    i = complex<double>(0.0, 1.0);
    edgeZero = QMDDEdge(.0, nullptr);
    edgeOne = QMDDEdge(1.0, nullptr);
}