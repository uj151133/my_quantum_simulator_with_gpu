#include "node.hpp">

int add_c(int x, int y){
    return x + y;
}

PYBIND11_MODULE(node, n) {
    n.def("add", &add_c);
}