#include "secaN11.hpp"

void secaN11() {
    QuantumCircuit q(11);

    array<int, 11> c;

    q.addZ(0);
    q.addH(0);

    q.addBarrier();
    q.addCX(0, 3);
    q.addCX(0, 6);
    q.addCZ(0, 3);
    q.addCZ(0, 6);

    q.addH(0);
    q.addH(3);
    q.addH(6);

    q.addZ(0);
    q.addZ(3);
    q.addZ(6);

    q.addCX(0, 1);
    q.addCX(0, 2);
    q.addCX(3, 4);
    q.addCX(3, 5);
    q.addCX(6, 7);
    q.addCX(6, 8);
    q.addCZ(0, 1);
    q.addCZ(0, 2);
    q.addCZ(3, 4);
    q.addCZ(3, 5);
    q.addCZ(6, 7);
    q.addCZ(6, 8);

    q.addBarrier();
    q.addH(9);
    q.addCX(9, 10);

    q.addBarrier();
    q.addCX(0, 9);
    c[9] = q.measure(9);
    q.addH(0);
    q.addCX(9, 10);
    c[0] = q.measure(0);
    q.addCZ(0, 10);

    q.addBarrier();
    q.addCX(10, 1);
    q.addCX(10, 2);
    q.addCX(3, 4);
    q.addCX(3, 5);
    q.addCX(6, 7);
    q.addCX(6, 8);
    q.addCZ(10, 1);
    q.addCZ(10, 2);
    q.addCZ(3, 4);
    q.addCZ(3, 5);
    q.addCZ(6, 7);
    q.addCZ(6, 8);
    q.addToff({1, 2}, 10);
    q.addToff({4, 5}, 3);
    q.addToff({7, 8}, 6);

    q.addBarrier();
    q.addH(10);
    q.addToff({1, 2}, 10);
    q.addH(10);
    q.addH(3);
    q.addToff({4, 5}, 3);
    q.addH(3);
    q.addH(6);
    q.addToff({7, 8}, 6);
    q.addH(6);

    q.addBarrier();
    q.addH(10);
    q.addH(3);
    q.addH(6);
    q.addZ(10);
    q.addZ(3);
    q.addZ(6);
    q.addCX(10, 3);
    q.addCX(10, 6);
    q.addCZ(10, 3);
    q.addCZ(10, 6);
    q.addToff({3, 6}, 10);
    q.addH(10);
    q.addToff({3, 6}, 10);
    q.addH(10);

    q.addBarrier();
    q.addH(10);
    q.addZ(10);
    c[10] = q.measure(10);
}