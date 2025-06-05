#include <iostream>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include <random>

#include "src/common/config.hpp"
// #include "src/common/jniUtils.hpp"
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

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 2.0 * M_PI);

    // random2(numQubits, numGates);

    // QuantumCircuit circuit(numQubits);
    // for ([[maybe_unused]] int _ = 0; _ < numGates; ++_) {
    //     double randomAngle = dis(gen);
    //     // circuit.addP(numQubits - 1, randomAngle);
    //     circuit.addRz(numQubits - 1, randomAngle);
    // }
    // // circuit.addToff({0, 1}, 3);
    // circuit.simulate();

    randomRotate4(numQubits, numGates);
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


    // int n_threads = std::thread::hardware_concurrency();

    // JNIEnv* env = nullptr;
    // if (!initJvm("./src/java", "./src/java/caffeine-3.2.0.jar", &env)) {
    //     std::cerr << "JVM起動失敗" << std::endl;
    //     return 1;
    // }
    // std::cout << "Main thread ID: " << std::this_thread::get_id() << std::endl;



    // std::cout << "Total unique threads used: " << threadIds.size() << std::endl;


    measureExecutionTime(execute);

    // threadPool.join();

    // detachJni();

    // detachJniForAllThreads();

    // if (env && g_OperationCache_cls) {
    //     env->DeleteGlobalRef(g_OperationCache_cls);
    //     g_OperationCache_cls = nullptr;
    // }

    return 0;
}

