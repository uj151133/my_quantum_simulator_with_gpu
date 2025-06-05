#ifndef RANDOMROTATE_HPP
#define RANDOMROTATE_HPP


#include <random>
#include "../../models/state.hpp"
#include "../../common/mathUtils.hpp"
#include "../../models/circuit.hpp"

using namespace std;

void randomRotate(size_t numQubits, size_t numGates);
void randomRotate2(size_t numQubits, size_t numGates);
void randomRotate4(size_t numQubits, size_t numGates);
void randomRotateDeep(size_t numQubits, size_t numLayers);

void random2(size_t numQubits, size_t numGates);
void random4(size_t numQubits, size_t numGates);

#endif