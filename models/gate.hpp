#ifndef GATE_HPP
#define GATE_HPP


#include <ginac/ginac.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

using namespace std;
using namespace GiNaC;


extern const matrix I_GATE;
extern const matrix X_GATE;
extern const matrix Y_GATE;
extern const matrix Z_GATE;
extern const matrix HADAMARD_GATE;
extern const matrix S_GATE;
extern const matrix S_DAGGER_GATE;
extern const matrix T_GATE;
extern const matrix T_DAGGER_GATE;
extern const matrix CNOT_GATE;
extern const matrix CZ_GATE;
extern const matrix TOFFOLI_GATE;
extern const matrix SWAP_GATE;

matrix RotateX(const ex &theta);
matrix RotateY(const ex &theta);
matrix RotateZ(const ex &theta);

matrix U1(const ex &lambda);
matrix U2(const ex &phi, const ex &lambda);
matrix U3(const ex &theta, const ex &phi, const ex &lambda);

// vector<vector<ex>> Ry(const ex &theta);

#endif