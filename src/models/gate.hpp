#ifndef GATE_HPP
#define GATE_HPP

#include "qmdd.hpp"
#include <ginac/ginac.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


using namespace std;
using namespace GiNaC;


namespace gate {
    extern const QMDDGate H_GATE;
    extern const QMDDGate I_GATE;
    extern const QMDDGate X_GATE;
}

QMDDGate createHGate();
QMDDGate createIGate();
QMDDGate createXGate();
QMDDGate createPlusXGate();
QMDDGate createMinusXGate();
QMDDGate createYGate();
QMDDGate createPlusYGate();
QMDDGate createMinusYGate();
QMDDGate createZGate();
QMDDGate createSGate();
QMDDGate createSDaggerGate();
QMDDGate createTGate();
QMDDGate createTDaggerGate();
extern const matrix CNOT_GATE;
extern const matrix CZ_GATE;
extern const matrix TOFFOLI_GATE;
extern const matrix SWAP_GATE;
QMDDGate createRotateXGate(double theta);

matrix RotateX(const ex &theta);
matrix RotateY(const ex &theta);
matrix RotateZ(const ex &theta);
matrix Rotate(const ex &k);

matrix U1(const ex &lambda);
matrix U2(const ex &phi, const ex &lambda);
matrix U3(const ex &theta, const ex &phi, const ex &lambda);

// vector<vector<ex>> Ry(const ex &theta);

#endif