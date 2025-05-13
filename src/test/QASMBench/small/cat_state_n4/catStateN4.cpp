#include "catStateN4.hpp"

void catStateN4() {
    QuantumCircuit bits(4);
    vector<int> c(4);
    bits.addH(0);
    bits.addCX(0, 1);
    bits.addCX(1, 2);
    bits.addCX(2, 3);
    c[0] = bits.read(0);
    c[1] = bits.read(1);
    c[2] = bits.read(2);
    c[3] = bits.read(3);
    return;
}
