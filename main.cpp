#include <iostream>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include <random>
#include <thread>
#include <signal.h>
#include <fstream>

#include "src/common/parameter.hpp"
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
#include "src/test/seca_n11/secaN11.hpp"
#include "src/translator/OpenQASM3/fallen.hpp"
#include "src/translator/OpenQASM3/gen/OpenQASM3Lexer.h"
#include "src/translator/OpenQASM3/gen/OpenQASM3Parser.h"

using namespace std;

// グローバルな共有メモリIPCサーバーインスタンス
IPC::SharedMemoryIPCServer* ipcServer = nullptr;

// シグナルハンドラ
void signalHandler(int) {
    if (ipcServer) ipcServer->stop();
    exit(0);
}

void execute() {
    int numQubits = 1;
    int qubitIndex = 1 - 1;

    QMDDEdge state = state::Ket0().getInitialEdge();
    for (int i = 1; i < numQubits; ++i) {
        state = mathUtils::kron(state::Ket0().getInitialEdge(), state);
    }


    vector<QMDDEdge> edges(qubitIndex, identityEdge);
    edges.push_back(gate::H().getInitialEdge());

    QMDDGate gate = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    });



    for ([[maybe_unused]] int _ = 0; _ < 100000; ++_) {
        state = mathUtils::mul(state, gate.getInitialEdge());
    }
}

bool translateAndExecuteQASM(const string& qasm_file) {
    try {
        cout << "Translating and executing QASM file: " << qasm_file << endl;

        ifstream file(qasm_file);
        if (!file.is_open()) {
            cerr << "Error: Cannot open file " << qasm_file << endl;
            return false;
        }
        string qasm_content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
        file.close();
        cout << "QASM content loaded successfully" << endl;

        antlr4::ANTLRInputStream input(qasm_content);
        OpenQASM3Lexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        OpenQASM3Parser parser(&tokens);

        OpenQASM3Parser::ProgramContext* tree = parser.program();

        CircuitGenerator generator;
        generator.visit(tree);

        string circuit_operations = generator.getCircuitCode();
        cout << "Translation completed. Generated operations:" << endl;
        cout << circuit_operations << endl;

        int max_qubit = generator.getMaxQubitIndex();
        int num_qubits = max_qubit + 1;
        if (num_qubits <= 0) {
            cerr << "Error: no qubits detected from QASM." << endl;
            return false;
        }

        cout << "Creating circuit with " << num_qubits << " qubits" << endl;
        QuantumCircuit circuit(num_qubits);

        generator.applyToCircuit(circuit);

        cout << "Starting simulation..." << endl;
        measureExecutionTime([&circuit]() { circuit.simulate(); });

        cout << "Simulation completed successfully!" << endl;
        return true;
    } catch (const exception& e) {
        cerr << "Error during translation and execution: " << e.what() << endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    // シグナルハンドラを設定
    signal(SIGINT,  signalHandler);
    signal(SIGTERM, signalHandler);

#ifdef __APPLE__
    const char* cfgPath = "/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml";
    const char* stgPath = "/Users/mitsuishikaito/my_quantum_simulator_with_gpu/setting.ini";
#elif __linux__
    const char* cfgPath = "/home/ark/my_quantum_simulator_with_gpu/config.yaml";
    const char* stgPath = "/home/ark/my_quantum_simulator_with_gpu/setting.ini";
#else
    #error "Unsupported operating system"
#endif

    // 設定ファイル読み込み（例外ガード）
    try {
        cout << "Loading config file: " << cfgPath << endl;
        cout << "Loading setting file: " << stgPath << endl;
        PARAMETER.loadFromFile(cfgPath, stgPath);
    } catch (const exception& e) {
        cerr << "Config load failed: " << e.what() << endl;
    }

    // コマンドライン引数を解析
    bool startSharedMemoryServer = false;
    string translateFile;
    int opt;
    // -t は引数必須なので「t:」
    while ((opt = getopt(argc, argv, "sht:")) != -1) {
        switch (opt) {
            case 's': startSharedMemoryServer = true; break;
            case 't': translateFile = optarg ? string(optarg) : string(); break;
            case 'h':
                cout << "Usage: " << argv[0] << " [-s] [-t file.qasm] [-h]\n";
                cout << "  -s: Start Shared Memory IPC server for GUI communication\n";
                cout << "  -t: Translate and run the given OpenQASM file\n";
                return 0;
            default:
                cout << "Use -h for help\n";
                return 1;
        }
    }
    // 旧形式 (-translate) 互換
    if (translateFile.empty() && argc > 2 && string(argv[1]) == "-translate") {
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
        cout << "Starting QMDD Simulator in standalone mode..." << endl;
        measureExecutionTime(execute);
        // execute();

        cout << "Total entries: " << UniqueTable::getInstance().getTotalEntryCount() << endl;
    }

    cout << "Program finished successfully." << endl;
    return 0;
}