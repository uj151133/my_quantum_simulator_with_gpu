#ifndef CALCULATION_H
#define CALCULATION_H

#include <ginac/ginac.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace GiNaC;

ex ComputeGCDOfElements(const matrix& matrix);

#endif // GATE_H