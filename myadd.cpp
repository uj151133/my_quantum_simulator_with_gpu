#include <pybind11/pybind11.h>

int add(int x, int y) {
    return x + y;
}

PYBIND11_MODULE(myadd, m) {
    m.def("add", &add);
}