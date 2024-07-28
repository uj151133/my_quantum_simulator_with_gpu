#include "iostream"
#include <ginac/ginac.h>
#include "src/models/bit.hpp"
#include "src/models/gate.hpp"
#include "src/models/uniquetable.hpp"
#include "src/models/qmdd.hpp"
#include "src/common/calculation.hpp"
#include "src/common/mathUtils.hpp"

using namespace GiNaC;

int main() {
    QMDDGate hGate = gate::H_GATE;
    cout << "hgate:" << hGate.getInitialEdge() << endl;
    QMDDGate xGate = gate::X_GATE;
    cout << "xgate:" << xGate.getInitialEdge() << endl;
    QMDDState ket0 = state::KET_0;
    auto result1 = mathUtils::addition(hGate.getInitialEdge(), ket0.getInitialEdge());
    cout << "result:" << result1 << endl;

    auto result2 = mathUtils::multiplication(hGate.getInitialEdge(), ket0.getInitialEdge());
    return 0;
}