#include "ipc_shared_memory.hpp"
#include "../models/circuit.hpp"
#include <iostream>
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
    std::cout << "SharedMemory IPC Server started" << std::endl;
    
    while (isRunning) {
        try {
            if (sharedData && sharedData->hasRequest) {
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
            SimulationResult errorResult;
            errorResult.success = false;
            errorResult.errorMessage = e.what();
            
            std::string response = serializeResult(errorResult);
            size_t responseSize = std::min(response.size(), sizeof(sharedData->responseData) - 1);
            std::memcpy(sharedData->responseData, response.c_str(), responseSize);
            sharedData->responseData[responseSize] = '\0';
            sharedData->responseSize = responseSize;
            sharedData->hasResponse = true;
            sharedData->hasRequest = false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void SharedMemoryIPCServer::stop() {
    isRunning = false;
}

SimulationResult SharedMemoryIPCServer::processCircuitRequest(const CircuitRequest& request) {
    SimulationResult result;
    
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
        circuit.simulate();
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
        
        result.success = true;
        result.executionTime = duration.count() / 1000.0;
        result.finalState = finalStateStream.str();
        
        std::cout << "=== Simulation Results ===" << std::endl;
        std::cout << "Success: " << (result.success ? "true" : "false") << std::endl;
        std::cout << "Execution time: " << result.executionTime << " ms" << std::endl;
        std::cout << "Final state info: " << result.finalState << std::endl;
        std::cout << "===========================" << std::endl;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = e.what();
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
        
        if (gateJson.contains("controlQubits")) {
            gate.control_qubits = gateJson["controlQubits"].get<std::vector<int>>();
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
