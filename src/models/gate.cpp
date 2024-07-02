#include "gate.hpp"

// #include "gate.cu"

#include <ginac/ginac.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>

using namespace std; 
using namespace GiNaC;
// namespace py = pybind11;

// const complex <int> i(0, 1);
// const ex sqrt2 = sqrt(ex(2));
// const ex i = I;

QMDDEdge createHGate() {
    complex<double> hWeight = 1.0 / sqrt(2.0);
    QMDDNode* hNode = new QMDDNode(4);

    hNode->edges[0] = QMDDEdge(1, nullptr);
    hNode->edges[1] = QMDDEdge(1, nullptr);
    hNode->edges[2] = QMDDEdge(1, nullptr);
    hNode->edges[3] = QMDDEdge(-1, nullptr);

    QMDDEdge hEdge(hWeight, hNode);

    return hEdge;
}

const matrix I_GATE = matrix{
    {1, 0},
    {0, 1}
};



const matrix X_GATE = matrix{
    {0, 1},
    {1, 0}
};

const matrix PLUS_X_GATE = matrix{
    {1 / sqrt(ex(2)), I / sqrt(ex(2))},
    {I / sqrt(ex(2)), 1 / sqrt(ex(2))}
};

const matrix MINUS_X_GATE = matrix{
    {1 / sqrt(ex(2)), -I / sqrt(ex(2))},
    {-I / sqrt(ex(2)), 1 / sqrt(ex(2))}
};

const matrix Y_GATE = matrix{ 
    {0, -I}, 
    {I, 0}
};

const matrix PLUS_Y_GATE = matrix{
    {1 / sqrt(ex(2)), 1 / sqrt(ex(2))},
    {-1 / sqrt(ex(2)), 1 / sqrt(ex(2))}
};

const matrix MINUS_Y_GATE = matrix{
    {1 / sqrt(ex(2)), -1 / sqrt(ex(2))},
    {1 / sqrt(ex(2)), 1 / sqrt(ex(2))}
};

const matrix Z_GATE = matrix{ 
    {1, 0},
    {0, -1}
};

const matrix H_GATE = matrix{
    {1 / sqrt(ex(2)), 1 / sqrt(ex(2))},
    {1 / sqrt(ex(2)), -1 / sqrt(ex(2))}
};

const matrix S_GATE = matrix{
    {1, 0},
    {0, I}
};

const matrix S_DAGGER_GATE = matrix{
    {1, 0},
    {0, -I}
};

const matrix T_GATE = matrix{
    {1, 0},
    {0, exp(I * ex(Pi / 4))}
};

const matrix T_DAGGER_GATE = matrix{
    {1, 0},
    {0, -exp(I * ex(Pi / 4))}
};

const matrix CNOT_GATE = matrix{
    {1, 0, 0, 0}, 
    {0, 1, 0, 0},
    {0, 0, 0, 1},
    {0, 0, 1, 0}
};

const matrix CZ_GATE = matrix{ 
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, -1}
};

const matrix TOFFOLI_GATE = matrix{ 
    {1, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0, 1, 0}
};

const matrix SWAP_GATE = matrix{ 
    {1, 0, 0, 0},
    {0, 0, 1, 0},
    {0, 1, 0, 0}, 
    {0, 0, 0, 1}
};

matrix RotateX(const ex &theta){
    return matrix{
        {cos(theta / 2), -I * sin(theta / 2)},
        {-I * sin(theta / 2), cos(theta / 2)}
    };
}

matrix RotateY(const ex &theta){
    return matrix{ 
        {cos(theta / 2), -sin(theta / 2)},
        {sin(theta / 2), cos(theta / 2)}
    };
}

matrix RotateZ(const ex &theta){
    return matrix{
        {exp(-I * theta / 2), 0},
        {0, exp(I * theta / 2)}
    };
}
matrix Rotate(const ex &k){
    return matrix{
        {1, 0},
        {0, exp((2 * Pi * I) / pow(2, k))}
    };
}

matrix U1(const ex &lambda){
    return matrix{
        {1, 0},
        {0, exp(I * lambda)}
    };
}

matrix U2(const ex &phi, const ex &lambda){
    return matrix{
        {1, -exp(I * lambda)},
        {exp(I * phi), exp(I * (lambda + phi))}
    };
}

matrix U3(const ex &theta, const ex &phi, const ex &lambda){
    return matrix{
        {cos(theta / 2), -exp(I * lambda) * sin(theta / 2)},
        {exp(I * phi) * sin(theta / 2), exp(I * (lambda + phi)) * cos(theta / 2)}
    };
}

// vector<vector<ex>> Ry(const ex &theta){
//     return {
//         {cos(theta / 2), -sin(theta / 2)},
//         {sin(theta / 2), cos(theta / 2)}
//         };
// }

PYBIND11_MODULE(gate, m) {

    m.attr("I_GATE") = I_GATE;

    m.attr("NOT_GATE") = X_GATE;

    m.attr("Z_GATE") = Z_GATE;

    m.attr("H_GATE") = H_GATE;

    m.attr("CNOT_GATE") = CNOT_GATE;

    m.attr("CZ_GATE") = CZ_GATE;

    m.attr("TOFFOLI_GATE") = TOFFOLI_GATE;

    m.attr("SWAP_GATE") = SWAP_GATE;

    m.def("RotateX", &RotateX);
    
    m.def("RotateY", &RotateY);

    m.def("RotateZ", &RotateZ);

    m.def("U1", &U1);

    m.def("U2", &U2);

    m.def("U3", &U3);

    // m.def("Ry", [](double theta, py::array_t<double> matrix) {
    //     double* ptr = static_cast<double*>(matrix.request().ptr);
    //     double* d_matrix;

    //     cudaMalloc(&d_matrix, 4 * sizeof(double));
    //     cudaMemcpy(d_matrix, ptr, 4 * sizeof(double), cudaMemcpyHostToDevice);

    //     Ry<<<1, 1>>>(theta, d_matrix);
    //     cudaDeviceSynchronize();

    //     cudaMemcpy(ptr, d_matrix, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    //     cudaFree(d_matrix);
    // });
}

