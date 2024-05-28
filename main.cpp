#include "iostream"
#include "myadd.hpp"
#include "gate.hpp"

int main() {
    int result = add(2, 3);
    std::cout << "Result of add(2, 3) is: " << result << std::endl;
    return 0;
}