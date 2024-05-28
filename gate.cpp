#include "gate.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <iostream>
#include <cmath> 
#include <complex>

using namespace std; 

// const complex <int> i(0, 1);

const vector<vector<int>> I_GATE = {
    {1, 0},
    {0, 1}
};


const vector<vector<int>> NOT_GATE = {
    {0, 1},
    {1, 0}
};

// const vector<vector<complex<int>>> Y_GATE = {
//     {0, -i},
//     {i, 0}
// };

const vector<vector<int>> Z_GATE = {
    {1, 0},
    {0, -1}
};

const vector<vector<double>> HADAMARD_GATE = {
    {1.0 / std::sqrt(2) * 1, 1.0 / std::sqrt(2) * 1},
    {1.0 / std::sqrt(2) * 1, 1.0 / std::sqrt(2) * -1}
};

const vector<vector<int>> CNOT_GATE = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1},
    {0, 0, 1, 0}
};

const vector<vector <int>> CZ_GATE = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, -1}
};

const vector<vector<int>> TOFFOLI_GATE = {
    {1, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0, 1, 0},
};

const vector<vector<int>> SWAP_GATE = {
    {1, 0, 0, 0},
    {0, 0, 1, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1}
};

vector<vector<double>> Ry(double theta){
    return {
        {cos(theta / 2), -sin(theta / 2)},
        {sin(theta / 2), cos(theta / 2)}
        };
}
PYBIND11_MODULE(gate, m) {

    m.attr("I_GATE") = I_GATE;

    m.attr("NOT_GATE") = NOT_GATE;

    m.attr("Z_GATE") = Z_GATE;

    m.attr("HADAMARD_GATE") = HADAMARD_GATE;

    m.attr("CNOT_GATE") = CNOT_GATE;

    m.attr("CZ_GATE") = CZ_GATE;

    m.attr("TOFFOLI_GATE") = TOFFOLI_GATE;

    m.attr("SWAP_GATE") = SWAP_GATE;

    m.def("Ry", &Ry);
}

