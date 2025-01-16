#ifndef GATE_HPP
#define GATE_HPP

#include <iostream>
#include <gsl/gsl_sf_bessel.h>
#include "qmdd.hpp"
#include "../common/constant.hpp"
#include "../common/mathUtils.hpp"
using namespace std;
namespace gate {
    /* Identity gate and global phase */
    QMDDGate I();
    QMDDGate Ph(double delta);

    /* Clifford qubit gates */
    QMDDGate X();
    QMDDGate Y();
    QMDDGate Z();
    QMDDGate S();
    QMDDGate Sdagger();
    QMDDGate V();
    QMDDGate Vdagger();
    QMDDGate H();
    QMDDGate CX1();
    QMDDGate CX2();
    QMDDGate varCX();
    QMDDGate CZ();
    QMDDGate DCNOT();
    QMDDGate SWAP();
    QMDDGate iSWAP();

    /* Non-Clifford qubit gates */
    QMDDGate P(double phi);
    QMDDGate T();
    QMDDGate Tdagger();
    QMDDGate CP(double phi);
    QMDDGate CS();

    /* Rotation operator gates */
    QMDDGate R(double theta, double phi);
    QMDDGate Rx(double theta);
    QMDDGate Ry(double theta);
    QMDDGate Rz(double theta);

    /* Two-qubit interaction gates */
    QMDDGate Rxx(double phi);
    QMDDGate Ryy(double phi);
    QMDDGate Rzz(double phi);
    QMDDGate Rxy(double phi);

    /* Non-Clifford swap gates */
    QMDDGate SquareSWAP();
    QMDDGate SquareiSWAP();
    QMDDGate SWAPalpha(double alpha);
    QMDDGate FREDKIN();

    /* Other named qubit */
    QMDDGate U(double theta, double phi, double lambda);
    QMDDGate U1(double theta);
    QMDDGate U2(double phi, double lambda);
    QMDDGate U3(double theta, double phi, double lamda);
    QMDDGate BARENCO(double alpha, double phi, double theta);
    QMDDGate B();
    QMDDGate CSX();
    QMDDGate N(double a, double b, double c);
    QMDDGate DB();
    QMDDGate ECR();
    QMDDGate fSim(double theta, double phi);
    QMDDGate G(double theta);
    QMDDGate M();
    QMDDGate syc();
    QMDDGate CZS(double theta, double phi, double gamma);
    QMDDGate D(double theta);
    QMDDGate RCCX();
    QMDDGate PG();
    QMDDGate Toff();
    QMDDGate fFredkin();
}



// QMDDGate createPlusYGate();
// QMDDGate createMinusYGate();
// QMDDGate createSDaggerGate();
// QMDDGate createTDaggerGate();


// matrix Rotate(const ex &k);

// matrix U1(const ex &lambda);
// matrix U2(const ex &phi, const ex &lambda);

// vector<vector<ex>> Ry(const ex &theta);

#endif