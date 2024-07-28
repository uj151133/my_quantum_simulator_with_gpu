#ifndef BIT_HPP
#define BIT_HPP

#include "qmdd.hpp"
#include <ginac/ginac.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
using namespace GiNaC;
namespace state {
    extern const QMDDState KET_0;
}
QMDDState setKet0();
extern const matrix KET_0;
extern const matrix KET_1;
extern const matrix KET_PLUS;
extern const matrix KET_MINUS;
extern const matrix BRA_0;
extern const matrix BRA_1;
extern const matrix BRA_PLUS;
extern const matrix BRA_MINUS;
#endif