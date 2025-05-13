#include "deutschN2.hpp"

void deutschN2() {
    QuantumCircuit q(2);
    vector<int> c(2);
    q.addX(1);
    q.addH(0);
    q.addH(1);
    q.addCX(0, 1);
    q.addH(0);
    c[0] = q.read(0);
    c[1] = q.read(1);
    return;
}
