#include <iostream>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include <random>

#include "src/common/config.hpp"
#include "src/models/qmdd.hpp"
#include "src/common/constant.hpp"
#include "src/models/gate.hpp"
#include "src/models/state.hpp"
#include "src/models/uniqueTable.hpp"
#include "src/common/mathUtils.hpp"
#include "src/common/calculation.hpp"
#include "src/models/circuit.hpp"
#include "src/common/monitor.hpp"
#include "src/test/Grover/grover.hpp"
#include "src/test/random/randomRotate.hpp"

using namespace std;


void execute() {


    // OperationCache& cache = OperationCache::getInstance();

    int numQubits = 10;
    int numGates = 200;

    randomRotate(numQubits, numGates);

    // QuantumCircuit circuit(numQubits);
    // for ([[maybe_unused]] int _ = 0; _ < numGates; ++_) {
    //     double randomAngle = dis(gen);
    //     // circuit.addP(numQubits - 1, randomAngle);
    //     circuit.addRz(numQubits - 1, randomAngle);
    // }
    // // circuit.addToff({0, 1}, 3);
    // circuit.simulate();

    // randomRotate4(numQubits, numGates);
    // int omega = std::pow(2, numQubits) - 1;

    // grover(numQubits, omega);
    
    // UniqueTable::getInstance().printAllEntries();

}



int main() {

    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif


    measureExecutionTime(execute);
    OperationCacheClient::getInstance().cleanup();

    return 0;
}

