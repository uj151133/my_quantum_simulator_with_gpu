#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cstring>
#include <atomic>

// 型定義
struct GateCommand {
    std::string type;
    std::vector<int> qubits;
    std::vector<int> control_qubits;
    double angle = 0.0;
};

struct CircuitRequest {
    int num_qubits;
    std::vector<GateCommand> gates;
};

struct SimulationResult {
    bool success = false;
    std::string errorMessage;
    double executionTime = 0.0;
    std::string finalState;
    std::string simulationLog; // C++シミュレーションの詳細ログ
};

// 共有メモリ構造体
struct SharedMemoryData {
    bool hasRequest = false;
    bool hasResponse = false;
    char requestData[8192];
    char responseData[8192];
    size_t requestSize = 0;
    size_t responseSize = 0;
};

namespace IPC {
    class SharedMemoryIPCServer {
    private:
        std::atomic<bool> isRunning{false};
        SharedMemoryData* sharedData = nullptr;
        
    public:
        SharedMemoryIPCServer();
        ~SharedMemoryIPCServer();
        
        bool initialize();
        void run();
        void stop();
        
        SimulationResult processCircuitRequest(const CircuitRequest& request);
        CircuitRequest parseRequest(const std::string& jsonRequest);
        std::string serializeResult(const SimulationResult& result);
    };

    class SharedMemoryIPCClient {
    private:
        SharedMemoryData* sharedData = nullptr;
        
    public:
        SharedMemoryIPCClient();
        ~SharedMemoryIPCClient();
        
        bool initialize();
        SimulationResult sendRequest(const CircuitRequest& request);
        
    private:
        std::string serializeRequest(const CircuitRequest& request);
        SimulationResult parseResponse(const std::string& jsonResponse);
    };
}
