#include <iostream>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <getopt.h>


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

    int numQubits = 3;
    int numGates = 200;

    QuantumCircuit circuit(numQubits);
    for ([[maybe_unused]] int _ = 0; _ < numGates; ++_) {
        circuit.addAllX();
    }
    // circuit.addToff({0, 1}, 3);
    // circuit.simulate();

    // randomRotate(numQubits, numGates);
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

    try {
        cout << "Initializing OperationCacheClient..." << endl;
        OperationCacheClient::initialize("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/src/java/liboperation-cache.dylib");
        
        cout << "Getting instance..." << endl;
        OperationCacheClient& client = OperationCacheClient::getInstance();
        
        cout << "Client instance obtained successfully" << endl;
        
        // size()の呼び出しを有効化してテスト
        int64_t cache_size = client.size();
        cout << "Cache size: " << cache_size << endl;
        
        cout << "Program completed successfully" << endl;
        
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error occurred" << endl;
        return 1;
    }

    // int n_threads = std::thread::hardware_concurrency();

    // std::cout << "Main thread ID: " << std::this_thread::get_id() << std::endl;



    // std::cout << "Total unique threads used: " << threadIds.size() << std::endl;


    // measureExecutionTime(execute);

    // threadPool.join();


    return 0;
}

