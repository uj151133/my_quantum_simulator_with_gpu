#ifndef GATE_HPP
#define GATE_HPP

#include <vector>
#include <cmath> 
#include <complex>


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

// extern const complex<int> i;
extern const vector<vector<int>> I_GATE;
extern const vector<vector<int>> NOT_GATE;
// extern const vector<vector<complex<int>>> Y_GATE;
extern const vector<vector<int>> Z_GATE;
extern const vector<vector<double>> HADAMARD_GATE;
extern const vector<vector<int>> CNOT_GATE;
extern const vector<vector<int>> CZ_GATE;
extern const vector<vector<int>> TOFFOLI_GATE;
extern const vector<vector<int>> SWAP_GATE;

vector<vector<double>> Ry(double theta);

#endif // GATE_HPP