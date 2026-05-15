#ifndef STATE_HPP
#define STATE_HPP

#include "qmdd.hpp"
#include "../common/bigBang.hpp"

using namespace std;

namespace state {
    QMDDSuite Ket0();
    QMDDSuite Ket1();
    QMDDSuite KetPlus();
    QMDDSuite KetMinus();
    QMDDSuite KetI();
    QMDDSuite KetIMinus();

    QMDDSuite Bra0();
    QMDDSuite Bra1();
    QMDDSuite BraPlus();
    QMDDSuite BraMinus();
    QMDDSuite BraI();
    QMDDSuite BraIMinus();
}
#endif