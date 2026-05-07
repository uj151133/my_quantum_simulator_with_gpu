#ifndef GATE_HPP
#define GATE_HPP

// #include <Eigen/Dense>
// #include <xsimd/xsimd.hpp>
#include <iostream>
#include <ostream>
#include <gsl/gsl_sf_bessel.h>
#include "qmdd.hpp"
#include "../common/mathUtils.hpp"
#include "../common/bigBang.hpp"

using namespace std;

enum class Type {
    I,
    Ph,
    X,
    Y,
    Z,
    S,
    Sdg,
    V,
    Vdg,
    H,
    CX,
    varCX,
    CY,
    CZ,
    SWAP,
    P,
    T,
    Tdg,
    CP,
    CS,
    CH,
    R,
    Rx,
    Ry,
    Rz,
    Rxx,
    Ryy,
    Rzz,
    Rxy,
    U,
    U1,
    U2,
    U3,
    CRx,
    CRy,
    CRz,
    CU,
    Toff,
    Other,
    VOID,
    ANKER,
    BAN,
    JOKER
};

ostream& operator<<(ostream& os, Type type);
string toString(Type type);
namespace gate {
    /* Identity gate and global phase */
    QMDDSuite I();
    QMDDSuite Ph(double delta);

    /* Clifford qubit gates */
    QMDDSuite X();
    QMDDSuite Y();
    QMDDSuite Z();
    QMDDSuite S();
    QMDDSuite Sdg();
    QMDDSuite V();
    QMDDSuite Vdg();
    QMDDSuite H();
    QMDDSuite CX1();
    QMDDSuite CX2();
    QMDDSuite varCX();
    QMDDSuite CZ();
    QMDDSuite DCNOT();
    QMDDSuite SWAP();
    QMDDSuite iSWAP();

    /* Non-Clifford qubit gates */
    QMDDSuite P(double phi);
    QMDDSuite T();
    QMDDSuite Tdg();
    QMDDSuite CP(double phi);
    QMDDSuite CS();

    /* Rotation operator gates */
    QMDDSuite R(double theta, double phi);
    QMDDSuite Rx(double theta);
    QMDDSuite Ry(double theta);
    QMDDSuite Rz(double theta);
    QMDDSuite Rk(int k);

    /* Two-qubit interaction gates */
    QMDDSuite Rxx(double phi);
    QMDDSuite Ryy(double phi);
    QMDDSuite Rzz(double phi);
    QMDDSuite Rxy(double phi);

    /* Non-Clifford swap gates */
    QMDDSuite SquareSWAP();
    QMDDSuite SquareiSWAP();
    QMDDSuite SWAPalpha(double alpha);
    QMDDSuite FREDKIN();

    /* Other named qubit */
    QMDDSuite U(double theta, double phi, double lambda);
    QMDDSuite U1(double theta);
    QMDDSuite U2(double phi, double lambda);
    QMDDSuite U3(double theta, double phi, double lamda);
    QMDDSuite BARENCO(double alpha, double phi, double theta);
    QMDDSuite B();
    QMDDSuite CSX();
    QMDDSuite N(double a, double b, double c);
    QMDDSuite DB();
    QMDDSuite ECR();
    QMDDSuite fSim(double theta, double phi);
    QMDDSuite G(double theta);
    QMDDSuite M();
    QMDDSuite syc();
    QMDDSuite CZS(double theta, double phi, double gamma);
    QMDDSuite D(double theta);
    QMDDSuite RCCX();
    QMDDSuite PG();
    QMDDSuite Toff();
    QMDDSuite fFredkin();
}



// QMDDSuite createPlusYGate();
// QMDDSuite createMinusYGate();
// QMDDSuite createSdgGate();
// QMDDSuite createTdgGate();


// matrix Rotate(const ex &k);

// matrix U1(const ex &lambda);
// matrix U2(const ex &phi, const ex &lambda);

// vector<vector<ex>> Ry(const ex &theta);

#endif