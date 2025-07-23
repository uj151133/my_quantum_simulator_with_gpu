#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::cout << "QMDD Simulator IPC Client Test" << std::endl;
    
    // Unix domain socketを作成
    int clientSocket = socket(AF_UNIX, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, "/tmp/qmdd_sim_pipe", sizeof(addr.sun_path) - 1);

    // サーバーに接続
    if (connect(clientSocket, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        std::cerr << "Failed to connect to server. Make sure qmdd_sim -i is running." << std::endl;
        close(clientSocket);
        return 1;
    }

    // テスト用の回路リクエストを作成
    json request;
    request["numQubits"] = 2;
    request["gates"] = json::array();
    
    // H gate on qubit 0
    json hGate;
    hGate["type"] = "H";
    hGate["qubits"] = json::array({0});
    request["gates"].push_back(hGate);
    
    // CNOT gate (control: 0, target: 1)
    json cnotGate;
    cnotGate["type"] = "CNOT";
    cnotGate["control_qubits"] = json::array({0});
    cnotGate["qubits"] = json::array({1});
    request["gates"].push_back(cnotGate);

    std::string requestStr = request.dump();
    std::cout << "Sending request: " << requestStr << std::endl;

    // リクエストを送信
    send(clientSocket, requestStr.c_str(), requestStr.length(), 0);

    // レスポンスを受信
    char buffer[4096];
    ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
    if (bytesRead > 0) {
        buffer[bytesRead] = '\0';
        std::cout << "Received response: " << buffer << std::endl;
        
        try {
            json response = json::parse(buffer);
            if (response["success"].get<bool>()) {
                std::cout << "Simulation successful!" << std::endl;
                std::cout << "Execution time: " << response["executionTime"].get<double>() << "ms" << std::endl;
            } else {
                std::cout << "Simulation failed: " << response["errorMessage"].get<std::string>() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "Failed to parse response: " << e.what() << std::endl;
        }
    }

    close(clientSocket);
    return 0;
}
