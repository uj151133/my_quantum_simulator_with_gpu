#ifndef GATE_CUH
#define GATE_CUH

#include "qmdd.cuh"

#include <cuComplex.h>
#include <math_constants.h>
#include <pybind11/pybind11.h>

using namespase std;

namespace py = pybind11;


__device__ extern cuDoubleComplex i;

__global__ void createIdentityGate(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createGlobalPhaseGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double delta);
__global__ void createPauliXGate(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createPauliYGate(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createPauliZGate(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createPhaseSGate(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createSquareRootOfXGate(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createHadamardGate(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createPhaseShiftGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi);
__global__ void createPhaseTGate(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createRotationAboutXGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta);
__global__ void createRotationAboutYGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta);
__global__ void createRotationAboutZGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta);
__global__ void createXXInteractionGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi);
__global__ void createYYInteractionGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi);
__global__ void createZZInteractionGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi);
__global__ void createXXPlusYYGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi);
__global__ void createGeneralSingleQubitRotationGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta, double phi, double lambda);
__global__ void createBarencoGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double alpha, double phi, double theta);
__global__ void createBerkeleyBGate(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createCoreEntanglingGate(cuDoubleComplex* weights, cuDoubleComplex** nodes, double a, double b, double c);

namespace gate {
    /* Identity gate and global phase */
    QMDDGate I();
    QMDDGate Ph(double delta);

    /* Clofford qubit gates*/
    QMDDGate X();
    QMDDGate Y();
    QMDDGate Z();
    QMDDGate S();
    QMDDGate V();
    QMDDGate H();

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

#endif // GATE_CUH
