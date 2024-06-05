#ifndef CALCULATION_H
#define CLCULATION_H

#include <ginac/ginac.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace GiNaC;

// 二つの式の共通因数を計算する関数
ex common_factor(const ex &expr1, const ex &expr2);

// Pythonバインディング用のラッパー関数
py::object common_factor_py(py::object py_expr1, py::object py_expr2);

#endif
