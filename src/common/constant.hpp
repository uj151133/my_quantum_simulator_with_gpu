#pragma once
#ifndef CONSTANT_HPP
#define CONSTANT_HPP
#include <complex>
#include <mutex>
#include "../models/qmdd.hpp"

using namespace std;

extern complex<double> i;
extern QMDDEdge edgeZero;
extern QMDDEdge edgeOne;
extern QMDDEdge identityEdge;
extern QMDDEdge braketZero;
extern QMDDEdge braketOne;
extern once_flag initEdgeFlag;
extern once_flag initExtendedEdgeFlag;

void initEdge();
void initExtendedEdge();

#endif