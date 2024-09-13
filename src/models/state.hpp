#ifndef STATE_HPP
#define STATE_HPP

#include "qmdd.hpp"

using namespace std;

namespace state {
    QMDDState Ket0();
    QMDDState Ket1();
    QMDDState KetPlus();
    QMDDState KetMinus();

    QMDDState Bra0();
    QMDDState Bra1();
    QMDDState BraPlus();
    QMDDState BraMinus();
}
#endif