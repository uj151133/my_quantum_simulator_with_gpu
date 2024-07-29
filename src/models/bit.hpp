#ifndef BIT_HPP
#define BIT_HPP

#include "qmdd.hpp"
#include <ginac/ginac.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
using namespace GiNaC;
namespace state {
    QMDDState KET_0();
    QMDDState KET_1();
    QMDDState KET_PLUS();
    QMDDState KET_MINUS();
}
#endif