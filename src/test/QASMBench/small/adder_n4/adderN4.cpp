#include "adderN4.hpp"

void adderN4() {
    QuantumCircuit q(4);
    vector<int> c(4);
    q.addX(0);
    q.addX(1);
    q.addH(3);
    q.addCX(2, 3);
    q.addT(0);
    q.addT(1);
    q.addT(2);
    q.addTdg(3);
    q.addCX(0, 1);
    q.addCX(2, 3);
    q.addCX(3, 0);
    q.addCX(1, 2);
    q.addCX(0, 1);
    q.addCX(2, 3);
    q.addTdg(0);
    q.addTdg(1);
    q.addTdg(2);
    q.addT(3);
    q.addCX(0, 1);
    q.addCX(2, 3);
    q.addS(3);
    q.addCX(3, 0);
    q.addH(3);
    c[0] = q.read(0);
    c[1] = q.read(1);
    c[2] = q.read(2);
    c[3] = q.read(3);
    return;
}
