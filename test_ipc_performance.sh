#!/bin/bash

# IPC性能テストスクリプト

echo "IPC Performance Comparison Test"
echo "================================"

# コンパイル
echo "Building test programs..."
cd /Users/mitsuishikaito/my_quantum_simulator_with_gpu

# Unix Domain Socket版のテスト（削除）
# echo "Building socket test..."

# 共有メモリ版のテストクライアント
cat > test_shm_client.cpp << 'EOF'
#include "src/common/ipc.hpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "Shared Memory IPC Client Test" << std::endl;
    
    IPC::SharedMemoryIPCClient client;
    if (!client.connect()) {
        std::cerr << "Failed to connect to shared memory server" << std::endl;
        return 1;
    }

    if (!client.isServerAvailable()) {
        std::cerr << "Server is not available" << std::endl;
        return 1;
    }

    // テスト用回路の作成
    IPC::CircuitRequest request;
    request.numQubits = 2;
    
    IPC::GateCommand hGate;
    hGate.type = "H";
    hGate.qubits = {0};
    request.gates.push_back(hGate);
    
    IPC::GateCommand cnotGate;
    cnotGate.type = "CNOT";
    cnotGate.control_qubits = {0};
    cnotGate.qubits = {1};
    request.gates.push_back(cnotGate);

    // 性能測定
    auto start = std::chrono::high_resolution_clock::now();
    
    IPC::SimulationResult result = client.sendRequest(request);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (result.success) {
        std::cout << "Simulation successful!" << std::endl;
        std::cout << "Total latency: " << duration.count() << " μs" << std::endl;
        std::cout << "Execution time: " << result.executionTime << " ms" << std::endl;
    } else {
        std::cout << "Simulation failed: " << result.errorMessage << std::endl;
    }

    return 0;
}
EOF

g++ -std=c++20 -O3 -I/opt/homebrew/include -L/opt/homebrew/lib \
    test_shm_client.cpp src/common/ipc.cpp \
    -lnlohmann_json -o test_shm_client

echo "Test programs built successfully"
echo ""
echo "Performance Test Instructions:"
echo "=============================="
echo ""
echo "Shared Memory IPC test:"
echo "   Terminal 1: cd build && ./qmdd_sim -s"
echo "   Terminal 2: ./test_shm_client"
echo ""
echo "Expected performance:"
echo "- Shared Memory: ~1-10μs latency, ~10-50GB/s throughput"
