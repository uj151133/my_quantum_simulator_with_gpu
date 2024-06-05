#include <iostream>
#include <ginac/ginac.h>
#include <pybind11/pybind11.h>

using namespace GiNaC;
namespace py = pybind11;

// 二つの式の共通因数を計算する関数
ex common_factor(const ex &expr1, const ex &expr2) {
    return gcd(expr1, expr2);
}

// Pythonバインディング用のラッパー関数
py::object common_factor_py(py::object py_expr1, py::object py_expr2) {
    ex expr1 = py::cast<ex>(py_expr1);
    ex expr2 = py::cast<ex>(py_expr2);
    ex result = common_factor(expr1, expr2);
    return py::cast(result);
}

// Pythonモジュール定義
PYBIND11_MODULE(common_factor, m) {
    m.def("common_factor", &common_factor_py, "Calculate the common factor of two expressions");
}
