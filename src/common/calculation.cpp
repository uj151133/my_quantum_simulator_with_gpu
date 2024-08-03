#include "calculation.hpp"
// #include "gate.cu"

#include <ginac/ginac.h>>>

using namespace GiNaC;

ex ComputeGCDOfElements(const matrix& mat) {
    ex gcd_result = 0;

    for (unsigned i = 0; i < mat.rows(); ++i) {
        for (unsigned j = 0; j < mat.cols(); ++j) {
            ex elem = mat(i, j);
            if (elem != 0) {
                if (gcd_result == 0) {
                    gcd_result = elem;
                } else {
                    gcd_result = gcd(gcd_result, elem);
                }
            }
        }
    }
    if (gcd_result == 0) {
        return 0;
    }
    bool all_elements_contain_I = true;
    for (unsigned i = 0; i < mat.rows(); ++i) {
        for (unsigned j = 0; j < mat.cols(); ++j) {
            ex elem = mat(i, j);
            if (elem != 0 && !elem.has(I)) {
                all_elements_contain_I = false;
                break;
            }
        }
        if (!all_elements_contain_I) {
            break;
        }
    }
    if (all_elements_contain_I) {
        std::cout << "has I" << std::endl;
        gcd_result *= I;
    }
    return gcd_result;
}

PYBIND11_MODULE(calculation, m) {
    m.def("ComputeGCDOfElements", &ComputeGCDOfElements);
}