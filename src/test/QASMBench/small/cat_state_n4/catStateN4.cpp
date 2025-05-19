#include "catStateN4.hpp"

void catStateN4() {
    QuantumCircuit bits(4);
    vector<int> c(4);
    bits.addH(0);
    bits.addCX(0, 1);
    bits.addCX(1, 2);
    bits.addCX(2, 3);
    c[0] = bits.measure(0);
    c[1] = bits.measure(1);
    c[2] = bits.measure(2);
    c[3] = bits.measure(3);
    return;
}
