#include "iostream"
#include <ginac/ginac.h>
#include "src/models/bit.hpp"
#include "src/models/gate.hpp"

using namespace GiNaC;

int main() {
    matrix h = X_GATE;
    std::cout << "Result is: " << Y_GATE << std::endl;
    for (unsigned i = 0; i < X_GATE.rows(); ++i) {
        for (unsigned j = 0; j < X_GATE.cols(); ++j) {
            std::cout << X_GATE(i, j) << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}