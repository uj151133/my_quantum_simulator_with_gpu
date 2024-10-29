#ifndef CONSTANT_HPP
#define CONSTANT_HPP

#include "../models/qmdd.hpp"
using namespace std;

static complex<double> i(0.0, 1.0);
static const QMDDEdge edgeZero(.0, nullptr);
static const QMDDEdge edgeOne(1.0, nullptr);

#endif