#ifndef GATE_CUH
#define GATE_CUH

#include "qmdd.cuh"

#include <cuComplex.h>
#include <math_constants.h>>

using namespase std;

namespace py = pybind11;


__device__ extern cuDoubleComplex i;

__global__ void createZeroNode(QMDDNode* node);
__global__ void createIdentityNode(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createGlobalPhaseNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double delta);
__global__ void createPauliXNode(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createPauliYNode(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createPauliZNode(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createPhaseSNode(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createSquareRootOfXNode(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createHadamardNode(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createPhaseShiftNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi);
__global__ void createPhaseTNode(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createRotationAboutXNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta);
__global__ void createRotationAboutYNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta);
__global__ void createRotationAboutZNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta);
__global__ void createXXInteractionNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi);
__global__ void createYYInteractionNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi);
__global__ void createZZInteractionNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi);
__global__ void createXXPlusYYNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi);
__global__ void createGeneralSingleQubitRotationNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta, double phi, double lambda);
__global__ void createBarencoNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double alpha, double phi, double theta);
__global__ void createBerkeleyBNode(cuDoubleComplex* weights, cuDoubleComplex** nodes);
__global__ void createCoreEntanglingNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double a, double b, double c);

namespace gate {
    QMDDGate O();
    
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

#endif
