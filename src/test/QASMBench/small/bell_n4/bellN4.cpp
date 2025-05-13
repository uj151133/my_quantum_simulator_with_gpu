#include "bellN4.hpp"

void bellN4() {
    QuantumCircuit q(4);
    vector<int> m_b(1);
    vector<int> m_y(1);
    vector<int> m_a(1);
    vector<int> m_x(1);
    q.addH(0);
    q.addH(1);
    q.addH(3);
    q.addCX(0, 2);
    q.addU3(3, M_PI * 0.5, 0, M_PI * 0.75);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 0.25);
    q.addCX(3, 2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * 0.5);
    q.addCX(3, 2);
    q.addU3(3, M_PI * 0.5, M_PI * 0.5, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 0.75);
    q.addU3(0, M_PI * 0.5, 0, M_PI * 0.25);
    q.addCX(1, 0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * 0.5);
    q.addCX(1, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 0.5, M_PI * 1.0);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0);
    m_b[0] = q.read(2);
    m_y[0] = q.read(3);
    m_a[0] = q.read(0);
    m_x[0] = q.read(1);
    return;
}
