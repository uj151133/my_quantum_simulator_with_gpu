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

#include "src/translator/OpenQASM3/fallen.hpp"
#include "src/translator/OpenQASM3/gen/OpenQASM3Lexer.h"
#include "src/translator/OpenQASM3/gen/OpenQASM3Parser.h"

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


    shor(8);
}

bool translateAndExecuteQASM(const string& qasm_file) {
    try {
        cout << "Translating and executing QASM file: " << qasm_file << endl;
        
        ifstream file(qasm_file);
        if (!file.is_open()) {
            cerr << "Error: Cannot open file " << qasm_file << std::endl;
            return false;
        }
        
        string qasm_content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
        file.close();
        
        cout << "QASM content loaded successfully" << endl;
        
        antlr4::ANTLRInputStream input(qasm_content);
        OpenQASM3Lexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        OpenQASM3Parser parser(&tokens);
        
        // 3. パースツリーを取得
        OpenQASM3Parser::ProgramContext* tree = parser.program();
        
        // 4. 翻訳器を作成して訪問
        CircuitGenerator generator;
        generator.visit(tree);
        
        string circuit_operations = generator.getCircuitCode();
        
        cout << "Translation completed. Generated operations:" << endl;
        cout << circuit_operations << endl;
        
        int max_qubit = generator.getMaxQubitIndex();  // この関数を追加する必要がある
        int num_qubits = max_qubit + 1;

        cout << "Creating circuit with " << num_qubits << " qubits" << endl;
        QuantumCircuit circuit(num_qubits);

        generator.applyToCircuit(circuit);
        
        cout << "Starting simulation..." << endl;
        measureExecutionTime([&circuit]() {
            circuit.simulate();
        });
        
        cout << "Simulation completed successfully!" << endl;
        return true;
        
    } catch (const exception& e) {
        cerr << "Error during translation and execution: " << e.what() << endl;
        return false;
    }
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
    string translateFile = "";
    int opt;
    while ((opt = getopt(argc, argv, "sh")) != -1) {
        switch (opt) {
            case 's':
                startSharedMemoryServer = true;
                break;
            case 't':
                translateFile = optarg;
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

    if (argc > 1 && std::string(argv[1]) == "-translate") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " -translate <qasm_file>" << std::endl;
            return 1;
        }
        translateFile = argv[2];
    }

    if (!translateFile.empty()) {
        if (translateAndExecuteQASM(translateFile)) {
            cout << "Translation and execution completed successfully!" << endl;
            cout << "Total entries: " << UniqueTable::getInstance().getTotalEntryCount() << endl;
            return 0;
        } else {
            cerr << "Translation and execution failed!" << endl;
            return 1;
        }
    } else if (startSharedMemoryServer) {
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

