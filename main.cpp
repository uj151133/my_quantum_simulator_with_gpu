#include <iostream>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include <random>
#include <thread>
#include <signal.h>

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
#include "src/common/ipc_shared_memory.hpp"
#include "src/common/operationCacheClient.hpp"
#include "src/test/Shor/shor.hpp"

using namespace std;

// グローバルな共有メモリIPCサーバーインスタンス
IPC::SharedMemoryIPCServer* ipcServer = nullptr;

// シグナルハンドラ
void signalHandler(int signal) {
    if (ipcServer) {
        ipcServer->stop();
    }
    exit(0);
}


void execute() {


    // OperationCache& cache = OperationCache::getInstance();

    int numQubits = 13;
    int numGates = 200;

    // randomRotate(numQubits, numGates);

    shor(8);

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
    
    // UniqueTable::getInstance().printNodeNum();


}



int main(int argc, char* argv[]) {
    // シグナルハンドラを設定
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    #ifdef __APPLE__
        CONFIG.loadFromFile("/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml");
    #elif __linux__
        CONFIG.loadFromFile("/home/ark/my_quantum_simulator_with_gpu/config.yaml");
    #else
        #error "Unsupported operating system"
    #endif

    // コマンドライン引数を解析
    bool startSharedMemoryServer = false;
    int opt;
    while ((opt = getopt(argc, argv, "sh")) != -1) {
        switch (opt) {
            case 's':
                startSharedMemoryServer = true;
                break;
            case 'h':
                cout << "Usage: " << argv[0] << " [-s] [-h]" << endl;
                cout << "  -s: Start Shared Memory IPC server for GUI communication" << endl;
                cout << "  -h: Show this help message" << endl;
                return 0;
            default:
                cout << "Use -h for help" << endl;
                return 1;
        }
    }

    if (startSharedMemoryServer) {
        // 共有メモリIPCサーバーモードで起動
        cout << "Starting QMDD Simulator in Shared Memory IPC server mode..." << endl;
        
        ipcServer = new IPC::SharedMemoryIPCServer();
        if (ipcServer->initialize()) {
            cout << "Shared Memory IPC Server ready. Waiting for GUI connections..." << endl;
            ipcServer->run();
        } else {
            cout << "Failed to start Shared Memory IPC server" << endl;
            delete ipcServer;
            return 1;
        }
        
        delete ipcServer;
    } else {
        // 従来のシミュレーションモード
        cout << "Starting QMDD Simulator in standalone mode..." << endl;
        measureExecutionTime(execute);

        cout << "Total entries: " << UniqueTable::getInstance().getTotalEntryCount() << endl;
        
        // シミュレーション完了後にキャッシュをSQLiteに保存
        // cout << "Saving cache to SQLite database..." << endl;
        // auto& client = OperationCacheClient::getInstance();
        // client.saveCacheToSQLite();
    }

    // OperationCacheClient::getInstance().cleanup(); // ハングするためコメントアウト
    cout << "Program finished successfully." << endl;
    return 0;
}

