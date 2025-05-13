#include "basisChangeN3.hpp"

void basisChangeN3() {
    QuantumCircuit q(3);
    vector<int> c(3);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 0.0564006755);
    q.addU3(1, M_PI * 0.5, M_PI * 1.5, M_PI * 0.2945501109);
    q.addU3(0, M_PI * 0.5, M_PI * 1.5, M_PI * 1.5);
    q.addCZ(1, 2);
    q.addU3(2, M_PI * 0.1242949803, 0, 0);
    q.addU3(1, M_PI * 0.1242949803, M_PI * 0.5, M_PI * 1.5);
    q.addCZ(1, 2);
    q.addU3(2, M_PI * 0.0298311566, M_PI * 1.5, M_PI * 0.5);
    q.addU3(1, M_PI * 0.7273849664, M_PI * 1.5, M_PI * 1.0);
    q.addCZ(0, 1);
    q.addU3(1, M_PI * 0.328242091, 0, 0);
    q.addU3(0, M_PI * 0.328242091, M_PI * 0.5, M_PI * 1.5);
    q.addCZ(0, 1);
    q.addU3(1, M_PI * 0.1374475291, M_PI * 2.0, M_PI * 1.5);
    q.addU3(0, M_PI * 0.9766098537, 0, 0);
    q.addCZ(1, 2);
    q.addU3(2, M_PI * 0.2326621647, 0, 0);
    q.addU3(1, M_PI * 0.2326621647, M_PI * 0.5, M_PI * 1.5);
    q.addCZ(1, 2);
    q.addU3(2, M_PI * 0.5780153762, M_PI * 0.5, M_PI * 0.5);
    q.addU3(1, M_PI * 0.6257049652, M_PI * 0.5, 0);
    q.addCZ(0, 1);
    q.addU3(1, M_PI * 0.328242091, 0, 0);
    q.addU3(0, M_PI * 0.328242091, M_PI * 0.5, M_PI * 1.5);
    q.addCZ(0, 1);
    q.addU3(1, M_PI * 0.6817377913, 0, M_PI * 0.5);
    q.addU3(0, M_PI * 0.5, M_PI * 0.3593182384, M_PI * 1.5);
    q.addCZ(1, 2);
    q.addU3(2, M_PI * 0.1242949803, 0, 0);
    q.addU3(1, M_PI * 0.1242949803, M_PI * 0.5, M_PI * 1.5);
    q.addCZ(1, 2);
    q.addU3(2, M_PI * 0.5, M_PI * 1.3937948052, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.1556453697, M_PI * 0.5);
    c[0] = q.read(0);
    c[1] = q.read(1);
    c[2] = q.read(2);
    return;
}
