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
#include "src/common/bigBang.hpp"
#include "src/models/gate.hpp"
#include "src/models/state.hpp"
#include "src/models/uniqueTable.hpp"
#include "src/common/mathUtils.hpp"
#include "src/common/calculation.hpp"
#include "src/models/circuit.hpp"
#include "src/common/monitor.hpp"
#include "src/test/Grover/grover11MQT.hpp"
#include "src/test/QWalk/qwalk13MQT.hpp"
#include "src/test/random/randomRotate.hpp"
#include "src/common/ipc_shared_memory.hpp"
#include "src/common/operationCacheClient.hpp"
// #include "src/test/Shor/shor.hpp"
// #include "src/test/seca_n11/secaN11.hpp"
#include "src/translator/OpenQASM3/fallen.hpp"
#include "src/translator/OpenQASM3/gen/OpenQASM3Lexer.h"
#include "src/translator/OpenQASM3/gen/OpenQASM3Parser.h"
// #include "src/test/QAOA/qaoa16MQT.hpp"
#include "src/test/QWalk/qwalk13MQT.hpp"
#include "src/test/EfficientSU2Ansatz/efficientSU2Ansatz18MQT.hpp"
#include "src/test/QNN/qnn18MQT.hpp"
#include "src/test/RealAmplitudes/realAmplitudes18MQT.hpp"
#include "src/test/TwoLocalAnsatz/twoLocalAnsatz18MQT.hpp"
#include "src/test/QFTEntangled/qftEntangled18MQT.hpp"
#include "src/test/QPEInexact/qpeInexact18MQT.hpp"
#include "src/test/Shor/shor18MQT.hpp"
#include "src/test/QAOA/qaoa16MQT.hpp"
#include "src/test/AmplitudeEstimation/amplitudeEstimation18MQT.hpp"
#include "src/modules/threadPool.hpp"
#include "src/test/9symml/bench9symml.hpp"

using namespace std;

// グローバルな共有メモリIPCサーバーインスタンス
IPC::SharedMemoryIPCServer* ipcServer = nullptr;

// シグナルハンドラ
void signalHandler(int) {
    if (ipcServer) ipcServer->stop();
    exit(0);
}

void execute() {
    // grover11MQT();
    // qwalk13MQT();
    // qaoa16MQT();
    // amplitudeEstimation18MQT();
    // efficientSU2Ansatz18MQT();
    qnn18MQT();
    // realAmplitudes18MQT();
    // twoLocalAnsatz18MQT();
    // qftEntangled18MQT();
    // qpeInexact18MQT();
    // randomRotate(13, 200);
    // shor18MQT();
    // bench9symml();

    // randomCX(3, 5000, true);
    // Memo::getInstance().printAllEntries();
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

// #ifdef __APPLE__
//     const char* cfgPath = "/Users/mitsuishikaito/my_quantum_simulator_with_gpu/config.yaml";
//     const char* stgPath = "/Users/mitsuishikaito/my_quantum_simulator_with_gpu/setting.ini";
// #elif __linux__
//     const char* cfgPath = "/home/ark/my_quantum_simulator_with_gpu/config.yaml";
//     const char* stgPath = "/home/ark/my_quantum_simulator_with_gpu/setting.ini";
// #else
//     #error "Unsupported operating system"
// #endif

    // // 設定ファイル読み込み（例外ガード）
    // try {
    //     cout << "Loading config file: " << cfgPath << endl;
    //     cout << "Loading setting file: " << stgPath << endl;
    //     PARAMETER.loadFromFile(cfgPath, stgPath);
    // } catch (const exception& e) {
    //     cerr << "Config load failed: " << e.what() << endl;
    // }

    // fileUtils::resolveFilePath();
    // try {
    //     cout << "Loading config file: " << fileUtils::getConfigPath() << endl;
    //     cout << "Loading setting file: " << fileUtils::getSettingPath() << endl;
    //     PARAMETER.load(); // 一回だけ
    // } catch (const exception& e) {
    //     cerr << "Config load failed: " << e.what() << endl;
    // }

    birth();

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
    threadPool.shutdown();
    cout << "Program finished successfully." << endl;

    return 0;
}