#ifndef RANDOMROTATE_HPP
#define RANDOMROTATE_HPP


#include <random>
#include "../../models/state.hpp"
#include "../../common/mathUtils.hpp"
#include "../../models/circuit.hpp"

using namespace std;

void randomRotate(size_t numQubits, size_t numGates);

#endif