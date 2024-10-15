#ifndef STATE_CUH
#define STATE_CUH

#include "qmdd.cuh"

using namespase std;

namespace py = pybind11;

__global__ void createKet0Node(cuDoubleComplex* weights, QMDDNode* nodes);
__global__ void createKet1Node(cuDoubleComplex* weights, QMDDNode* nodes);
__global__ void createKetPlusNode(cuDoubleComplex* weights, QMDDNode* nodes);
__global__ void createKetMinusNode(cuDoubleComplex* weights, QMDDNode* nodes);

__global__ void createBra0Node(cuDoubleComplex* weights, QMDDNode* nodes);
__global__ void createBra1Node(cuDoubleComplex* weights, QMDDNode* nodes);
__global__ void createBraPlusNode(cuDoubleComplex* weights, QMDDNode* nodes);
__global__ void createBraMinusNode(cuDoubleComplex* weights, QMDDNode* nodes);

namespace state {
    QMDDState Ket0();
    QMDDState Ket1();
    QMDDState KetPlus();
    QMDDState KetMinus();

    QMDDState Bra0();
    QMDDState Bra1();
    QMDDState BraPlus();
    QMDDState BraMinus();
}
#endif