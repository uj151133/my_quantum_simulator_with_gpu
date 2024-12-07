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
extern once_flag initFlag;

void init();

#endif