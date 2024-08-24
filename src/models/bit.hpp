#ifndef BIT_HPP
#define BIT_HPP

#include "qmdd.hpp"

using namespace std;

namespace state {
    QMDDState KET_0();
    QMDDState KET_1();
    QMDDState KET_PLUS();
    QMDDState KET_MINUS();

    QMDDState BRA_0();
    QMDDState BRA_1();
    QMDDState BRA_PLUS();
    QMDDState BRA_MINUS();
}
#endif