#ifndef BIT_CUH
#define BIT_CUH

#include "qmdd.cuh"

using namespase std;

namespace py = pybind11;

namespace state {
    QMDDState KET_0();
    QMDDState KET_1();
    QMDDState KET_PLUS();
    QMDDState KET_MINUS();
}
#endif