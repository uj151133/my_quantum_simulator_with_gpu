#ifndef STATE_HPP
#define STATE_HPP

#include "qmdd.hpp"
#include "../common/constant.hpp"

using namespace std;

namespace state {
    QMDDState Ket0();
    QMDDState Ket1();
    QMDDState KetPlus();
    QMDDState KetMinus();
    QMDDState KetI();
    QMDDState KetIMinus();

    QMDDState Bra0();
    QMDDState Bra1();
    QMDDState BraPlus();
    QMDDState BraMinus();
    QMDDState BraI();
    QMDDState BraIMinus();
}
#endif