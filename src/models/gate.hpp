#ifndef GATE_HPP
#define GATE_HPP

#include "qmdd.hpp"
#include <ginac/ginac.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <functional>


using namespace std;
using namespace GiNaC;

namespace gate {
    QMDDGate ZERO();
    
    /* Identity gate and global phase */
    QMDDGate I();
    QMDDGate Ph(double delta);
    
    /* Clifford qubit gates */
    QMDDGate X();
    QMDDGate Y();
    QMDDGate Z();
    QMDDGate S();
    QMDDGate V();
    QMDDGate H();
    QMDDGate CX1();
    QMDDGate CX2();

    /* Non-Clifford qubit gates */
    QMDDGate P(double phi);
    QMDDGate T();  
    
    /* Rotation operator gates */
    QMDDGate Rx(double theta);
    QMDDGate Ry(double theta);
    QMDDGate Rz(double theta);

    /* Two-qubit interaction gates */
    QMDDGate Rxx(double phi); 
    QMDDGate Ryy(double phi);
    QMDDGate Rzz(double phi);
    QMDDGate Rxy(double phi);

    /*Other named qubit */
    QMDDGate U(double theta, double phi, double lambda);
    QMDDGate BARENCO(double alpha, double phi, double theta);
    QMDDGate B();
    QMDDGate N(double a, double b, double c);
}

// QMDDGate createPlusYGate();
// QMDDGate createMinusYGate();
// QMDDGate createSDaggerGate();
// QMDDGate createTDaggerGate();


// matrix Rotate(const ex &k);

matrix U1(const ex &lambda);
matrix U2(const ex &phi, const ex &lambda);

// vector<vector<ex>> Ry(const ex &theta);

#endif