#include "bit.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

using namespace std;

const vector<vector<int>>  KET_0 = {
    {1},
    {0}
};

const vector<vector<int>>  KET_1 = {
    {0},
    {1}
};

const vector<int>  BRA_0 = {1, 0};

const vector<int>  BRA_1 = {0, 1};


PYBIND11_MODULE(bit, m) {

    m.attr("KET_0") = KET_0;
    m.attr("KET_1") = KET_1;
    m.attr("BRA_0") = BRA_0;
    m.attr("BRA_1") = BRA_1;
}