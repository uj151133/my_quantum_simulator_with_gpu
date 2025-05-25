#include <iostream>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <getopt.h>


#include "src/common/config.hpp"
#include "src/common/jniUtils.hpp"
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


    try {
        // OperationCache& cache = OperationCache::getInstance();

        int numQubits = 5;
        // int numGates = 200;

        // randomRotate(numQubits, numGates);
        int omega = std::pow(2, numQubits) - 1;

        grover(numQubits, omega);

        
        // UniqueTable::getInstance().printAllEntries();
    } catch (const std::exception& e) {
        std::cerr << "Exception in execute(): " << e.what() << std::endl;
        throw; // 再スロー
    } catch (...) {
        std::cerr << "Unknown exception in execute()" << std::endl;
        throw; // 再スロー
    }

}



int main() {
try {
        #ifdef __APPLE__
            CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
        #elif __linux__
            CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
        #else
            #error "Unsupported operating system"
        #endif

        // measureExecutionTime関数の呼び出しを安全にラップ
        try {
            measureExecutionTime(execute);
        } catch (const std::future_error& e) {
            std::cerr << "Future error: " << e.what() << std::endl;
            std::cerr << "Error code: " << e.code() << std::endl;
            return 1;
        } catch (const std::exception& e) {
            std::cerr << "Exception during execution: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown fatal error" << std::endl;
        return 1;
    }
}

