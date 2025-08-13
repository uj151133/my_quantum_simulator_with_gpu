#include "ipc_shared_memory.hpp"
#include "../models/circuit.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <thread>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace IPC;

// SharedMemoryIPCServer implementation
SharedMemoryIPCServer::SharedMemoryIPCServer() {
    sharedData = new SharedMemoryData();
}

SharedMemoryIPCServer::~SharedMemoryIPCServer() {
    stop();
    delete sharedData;
}

bool SharedMemoryIPCServer::initialize() {
    return sharedData != nullptr;
}

void SharedMemoryIPCServer::run() {
    isRunning = true;
    std::cout << "SharedMemory IPC Server started (supports file-based and memory-based IPC)" << std::endl;
    
    // ファイルベースIPC用のディレクトリを作成
    // 固定パスを使用してGUIアプリケーションと確実に同期
    std::string tempDir = "/var/folders/zm/rwvnpn_j31q54p72tw6qfz_h0000gn/T/qmdd_ipc";
    
    std::cout << "Using fixed temp directory: " << tempDir << std::endl;
    
    // 固定パスでディレクトリを作成
    std::string createDirCommand = "mkdir -p \"" + tempDir + "\"";
    system(createDirCommand.c_str());
    
    std::string requestFile = tempDir + "/request.json";
    std::string responseFile = tempDir + "/response.json";
    std::string flagFile = tempDir + "/request_ready.flag";
    
    std::cout << "File-based IPC monitoring: " << tempDir << std::endl;
    
    while (isRunning) {
        try {
            // **ファイルベースIPCをチェック**
            if (std::ifstream(flagFile)) {
                std::cout << "🚩 File-based IPC request detected!" << std::endl;
                
                // リクエストファイルを読み取り
                std::ifstream requestStream(requestFile);
                if (requestStream.is_open()) {
                    std::string requestStr((std::istreambuf_iterator<char>(requestStream)),
                                         std::istreambuf_iterator<char>());
                    requestStream.close();
                    
                    std::cout << "📖 Read request from file (" << requestStr.length() << " bytes)" << std::endl;
                    std::cout << "📋 Request JSON: " << requestStr << std::endl;
                    
                    // フラグファイルを削除（処理中であることを示す）
                    std::remove(flagFile.c_str());
                    
                    // リクエストを処理
                    CircuitRequest circuitReq = parseRequest(requestStr);
                    SimulationResult result = processCircuitRequest(circuitReq);
                    
                    // レスポンスファイルに書き込み
                    std::string response = serializeResult(result);
                    std::ofstream responseStream(responseFile);
                    if (responseStream.is_open()) {
                        responseStream << response;
                        responseStream.close();
                        std::cout << "📝 Wrote response to file (" << response.length() << " bytes)" << std::endl;
                        std::cout << "📋 Response JSON: " << response << std::endl;
                    }
                } else {
                    std::cout << "❌ Could not read request file" << std::endl;
                    std::remove(flagFile.c_str()); // エラー時もフラグを削除
                }
            }
            
            // **メモリベースIPCもチェック（後方互換性のため）**
            if (sharedData && sharedData->hasRequest) {
                std::cout << "🧠 Memory-based IPC request detected!" << std::endl;
                std::string requestStr(sharedData->requestData);
                CircuitRequest circuitReq = parseRequest(requestStr);
                SimulationResult result = processCircuitRequest(circuitReq);
                
                std::string response = serializeResult(result);
                size_t responseSize = std::min(response.size(), sizeof(sharedData->responseData) - 1);
                std::memcpy(sharedData->responseData, response.c_str(), responseSize);
                sharedData->responseData[responseSize] = '\0';
                sharedData->responseSize = responseSize;
                sharedData->hasResponse = true;
                sharedData->hasRequest = false;
            }
        } catch (const std::exception& e) {
            std::cout << "❌ IPC Server error: " << e.what() << std::endl;
            
            SimulationResult errorResult;
            errorResult.success = false;
            errorResult.errorMessage = e.what();
            
            std::string response = serializeResult(errorResult);
            
            // ファイルベースのエラーレスポンス
            std::ofstream errorResponseStream(responseFile);
            if (errorResponseStream.is_open()) {
                errorResponseStream << response;
                errorResponseStream.close();
            }
            std::remove(flagFile.c_str()); // エラー時もフラグを削除
            
            // メモリベースのエラーレスポンス
            if (sharedData && sharedData->hasRequest) {
                size_t responseSize = std::min(response.size(), sizeof(sharedData->responseData) - 1);
                std::memcpy(sharedData->responseData, response.c_str(), responseSize);
                sharedData->responseData[responseSize] = '\0';
                sharedData->responseSize = responseSize;
                sharedData->hasResponse = true;
                sharedData->hasRequest = false;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // ファイルチェック間隔を100msに調整
    }
}

void SharedMemoryIPCServer::stop() {
    isRunning = false;
}

SimulationResult SharedMemoryIPCServer::processCircuitRequest(const CircuitRequest& request) {
    SimulationResult result;
    
    // **ログキャプチャ用の変数をtryブロック外で宣言**
    std::ostringstream logStream;
    std::streambuf* originalCoutBuffer = std::cout.rdbuf();
    
    try {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        std::cout << "=== QMDD Quantum Circuit Simulation Starting ===" << std::endl;
        std::cout << "Number of qubits: " << request.num_qubits << std::endl;
        std::cout << "Number of gates: " << request.gates.size() << std::endl;
        
        QuantumCircuit circuit(request.num_qubits);
        
        // ゲートを順番に追加し、詳細情報を出力
        for (size_t i = 0; i < request.gates.size(); i++) {
            const auto& gate = request.gates[i];
            std::cout << "Gate " << (i+1) << "/" << request.gates.size() 
                     << ": " << gate.type << " on qubit " << gate.qubits[0];
            
            if (!gate.control_qubits.empty()) {
                std::cout << " (control: " << gate.control_qubits[0] << ")";
            }
            if (gate.angle != 0.0) {
                std::cout << " (angle: " << gate.angle << " rad)";
            }
            std::cout << std::endl;
            
            // ゲートを追加し、対応する実際のメソッドを呼び出す
            if (gate.type == "H") {
                circuit.addH(gate.qubits[0]);
                std::cout << "  -> Called addH(" << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "X") {
                circuit.addX(gate.qubits[0]);
                std::cout << "  -> Called addX(" << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "Y") {
                circuit.addY(gate.qubits[0]);
                std::cout << "  -> Called addY(" << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "Z") {
                circuit.addZ(gate.qubits[0]);
                std::cout << "  -> Called addZ(" << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "I") {
                circuit.addI(gate.qubits[0]);
                std::cout << "  -> Called addI(" << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "T") {
                circuit.addT(gate.qubits[0]);
                std::cout << "  -> Called addT(" << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "Tdg" || gate.type == "T†") {
                circuit.addTdg(gate.qubits[0]);
                std::cout << "  -> Called addTdg(" << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "S") {
                circuit.addS(gate.qubits[0]);
                std::cout << "  -> Called addS(" << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "Sdg" || gate.type == "S†") {
                circuit.addSdg(gate.qubits[0]);
                std::cout << "  -> Called addSdg(" << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "P" && gate.angle != 0.0) {
                circuit.addP(gate.qubits[0], gate.angle);
                std::cout << "  -> Called addP(" << gate.qubits[0] << ", " << gate.angle << ")" << std::endl;
            } else if (gate.type == "RZ" && gate.angle != 0.0) {
                circuit.addRz(gate.qubits[0], gate.angle);
                std::cout << "  -> Called addRz(" << gate.qubits[0] << ", " << gate.angle << ")" << std::endl;
            } else if (gate.type == "RX" && gate.angle != 0.0) {
                circuit.addRx(gate.qubits[0], gate.angle);
                std::cout << "  -> Called addRx(" << gate.qubits[0] << ", " << gate.angle << ")" << std::endl;
            } else if (gate.type == "RY" && gate.angle != 0.0) {
                circuit.addRy(gate.qubits[0], gate.angle);
                std::cout << "  -> Called addRy(" << gate.qubits[0] << ", " << gate.angle << ")" << std::endl;
            } else if (gate.type == "CNOT" && gate.control_qubits.size() > 0) {
                circuit.addCX(gate.control_qubits[0], gate.qubits[0]);
                std::cout << "  -> Called addCX(" << gate.control_qubits[0] << ", " << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "CZ" && gate.control_qubits.size() > 0) {
                circuit.addCZ(gate.control_qubits[0], gate.qubits[0]);
                std::cout << "  -> Called addCZ(" << gate.control_qubits[0] << ", " << gate.qubits[0] << ")" << std::endl;
            } else if (gate.type == "Reset" || gate.type == "|0⟩") {
                circuit.addI(gate.qubits[0]); // リセットの代替実装
                std::cout << "  -> Called addI(" << gate.qubits[0] << ") [Reset simulation]" << std::endl;
            } else {
                std::cout << "  -> Unknown gate type: " << gate.type << std::endl;
            }
        }
        
        std::cout << "Starting QMDD simulation..." << std::endl;
        std::cout << "=== Circuit simulation output START ===" << std::endl;
        
        // stdout をキャプチャするために stringstream にリダイレクト
        std::ostringstream simulateLogCapture;
        std::streambuf* coutBuffer = std::cout.rdbuf();
        std::cout.rdbuf(simulateLogCapture.rdbuf());
        
        // simulate()メソッドを実行（このメソッドは内部でcoutを使用して詳細ログを出力）
        circuit.simulate();
        
        // stdoutを元に戻す
        std::cout.rdbuf(coutBuffer);
        
        // キャプチャしたログを表示
        std::string capturedSimulateLog = simulateLogCapture.str();
        std::cout << "Captured simulate() output:" << std::endl;
        std::cout << capturedSimulateLog << std::endl;
        
        std::cout << "=== Circuit simulation output END ===" << std::endl;
        std::cout << "QMDD simulation completed successfully!" << std::endl;
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        
        // 実際のfinalStateを取得
        auto finalState = circuit.getFinalState();
        std::stringstream finalStateStream;
        finalStateStream << "Final QMDD state computed with " << request.gates.size() 
                        << " gates on " << request.num_qubits << " qubits - "
                        << "Initial edge weight: " << finalState.getInitialEdge().weight
                        << ", Unique table key: " << finalState.getInitialEdge().uniqueTableKey;
        
        // 詳細なシミュレーションログを手動で生成（実際のsimulate()ログを含む）
        std::stringstream simulationLogBuilder;
        simulationLogBuilder << "QMDD Quantum Circuit Simulation Details:\n";
        simulationLogBuilder << "Number of qubits: " << request.num_qubits << "\n";
        simulationLogBuilder << "Number of gates: " << request.gates.size() << "\n\n";
        
        // 実際のsimulate()メソッドの出力を追加
        simulationLogBuilder << "=== simulate() method output ===\n";
        simulationLogBuilder << capturedSimulateLog;
        simulationLogBuilder << "=== End of simulate() output ===\n\n";
        
        // ゲートごとの詳細情報を生成（シミュレーション後の状態情報を含む）
        auto currentState = circuit.getFinalState();
        for (size_t i = 0; i < request.gates.size(); i++) {
            simulationLogBuilder << "Gate " << i << ": " << request.gates[i].type << " on qubit " << request.gates[i].qubits[0] << "\n";
            simulationLogBuilder << "  Weight: " << currentState.getInitialEdge().weight << "\n";
            simulationLogBuilder << "  Key: " << currentState.getInitialEdge().uniqueTableKey << "\n";
        }
        
        simulationLogBuilder << "Final state weight: " << finalState.getInitialEdge().weight << "\n";
        simulationLogBuilder << "Final state key: " << finalState.getInitialEdge().uniqueTableKey << "\n";
        
        result.success = true;
        result.executionTime = duration.count() / 1000.0;
        result.finalState = finalStateStream.str();
        result.simulationLog = simulationLogBuilder.str();
        
        // 結果を表示
        std::cout << "=== Simulation Results ===" << std::endl;
        std::cout << "Success: " << (result.success ? "true" : "false") << std::endl;
        std::cout << "Execution time: " << result.executionTime << " ms" << std::endl;
        std::cout << "Final state info: " << result.finalState << std::endl;
        std::cout << "===========================" << std::endl;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = e.what();
        result.simulationLog = "Error occurred during simulation: " + std::string(e.what());
        std::cout << "Simulation error: " << e.what() << std::endl;
    }
    
    return result;
}

CircuitRequest SharedMemoryIPCServer::parseRequest(const std::string& jsonRequest) {
    json j = json::parse(jsonRequest);
    CircuitRequest request;
    
    request.num_qubits = j["numQubits"];
    
    for (const auto& gateJson : j["gates"]) {
        GateCommand gate;
        gate.type = gateJson["type"];
        gate.qubits = gateJson["qubits"].get<std::vector<int>>();
        
        if (gateJson.contains("controlQubits") || gateJson.contains("control_qubits")) {
            if (gateJson.contains("controlQubits")) {
                gate.control_qubits = gateJson["controlQubits"].get<std::vector<int>>();
            } else {
                gate.control_qubits = gateJson["control_qubits"].get<std::vector<int>>();
            }
        }
        
        if (gateJson.contains("angle")) {
            gate.angle = gateJson["angle"];
        }
        
        request.gates.push_back(gate);
    }
    
    return request;
}

std::string SharedMemoryIPCServer::serializeResult(const SimulationResult& result) {
    json j;
    j["success"] = result.success;
    j["errorMessage"] = result.errorMessage;
    j["executionTime"] = result.executionTime;
    j["finalState"] = result.finalState;
    j["simulationLog"] = result.simulationLog; // C++シミュレーションログを追加
    
    return j.dump();
}

// SharedMemoryIPCClient implementation
SharedMemoryIPCClient::SharedMemoryIPCClient() {
    sharedData = new SharedMemoryData();
}

SharedMemoryIPCClient::~SharedMemoryIPCClient() {
    delete sharedData;
}

bool SharedMemoryIPCClient::initialize() {
    return sharedData != nullptr;
}

SimulationResult SharedMemoryIPCClient::sendRequest(const CircuitRequest& request) {
    SimulationResult result;
    
    try {
        std::string requestStr = serializeRequest(request);
        
        size_t requestSize = std::min(requestStr.size(), sizeof(sharedData->requestData) - 1);
        std::memcpy(sharedData->requestData, requestStr.c_str(), requestSize);
        sharedData->requestData[requestSize] = '\0';
        sharedData->requestSize = requestSize;
        sharedData->hasRequest = true;
        sharedData->hasResponse = false;
        
        while (!sharedData->hasResponse) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        std::string responseStr(sharedData->responseData);
        result = parseResponse(responseStr);
        
        sharedData->hasResponse = false;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = e.what();
    }
    
    return result;
}

std::string SharedMemoryIPCClient::serializeRequest(const CircuitRequest& request) {
    json j;
    j["numQubits"] = request.num_qubits;
    
    json gatesArray = json::array();
    for (const auto& gate : request.gates) {
        json gateJson;
        gateJson["type"] = gate.type;
        gateJson["qubits"] = gate.qubits;
        if (gate.angle != 0.0) {
            gateJson["angle"] = gate.angle;
        }
        if (!gate.control_qubits.empty()) {
            gateJson["controlQubits"] = gate.control_qubits;
        }
        gatesArray.push_back(gateJson);
    }
    
    j["gates"] = gatesArray;
    return j.dump();
}

SimulationResult SharedMemoryIPCClient::parseResponse(const std::string& jsonResponse) {
    SimulationResult result;
    
    try {
        json j = json::parse(jsonResponse);
        result.success = j["success"];
        result.errorMessage = j["errorMessage"];
        result.executionTime = j["executionTime"];
        result.finalState = j["finalState"];
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Failed to parse response: ") + e.what();
    }
    
    return result;
}
