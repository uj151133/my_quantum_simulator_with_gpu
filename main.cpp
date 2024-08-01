#include <iostream>
#include <cstdlib>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <mach/mach.h>
#include <ginac/ginac.h>
#include <yaml-cpp/yaml.h>
#include "src/models/bit.hpp"
#include "src/models/gate.hpp"
#include "src/models/uniqueTable.hpp"
#include "src/models/qmdd.hpp"
#include "src/common/calculation.hpp"
#include "src/common/mathUtils.hpp"

using namespace GiNaC;



void printMemoryUsage() {
    pid_t pid = getpid();
    std::string command = "ps -o rss= -p " + std::to_string(pid);
    
    // Create a pipe to capture the output of the command
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to run command.\n";
        return;
    }

    // Read the output of the command
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

    // Close the pipe and check for errors
    pclose(pipe);

    // Remove any trailing whitespace from the result
    result.erase(result.find_last_not_of(" \n\r\t") + 1);

    std::cout << "Memory usage: " << result << " KB" << std::endl;
}

void printMemoryUsageOnMac() {
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
        std::cerr << "Error getting memory info\n";
        return;
    }

    std::cout << "Memory usage on mac enviroment: " << info.resident_size / 1024 << " KB\n";
}

bool isExecuteGui() {
    YAML::Node config = YAML::LoadFile("config.yaml");
    bool guiEnabled = config["gui"]["enabled"].as<bool>();

    return guiEnabled;
}

int main() {
    printMemoryUsage();
    printMemoryUsageOnMac();

    bool isGuiEnabled = isExecuteGui();

    if (isGuiEnabled) {
        std::cout << "GUI is enabled." << std::endl;
    } else {
        std::cout << "GUI is disabled." << std::endl;
    }

    QMDDGate h1Gate = gate::H();
    cout << "h1gate:" << h1Gate.getInitialEdge() << endl;

    QMDDGate h2Gate = gate::H();
    cout << "h2gate:" << h2Gate.getInitialEdge() << endl;

    QMDDGate xGate = gate::X();
    cout << "xgate:" << xGate.getInitialEdge() << endl;
    QMDDState ket0 = state::KET_0();
    auto result1 = mathUtils::addition(h1Gate.getInitialEdge(), ket0.getInitialEdge());
    cout << "result:" << result1 << endl;

    // auto result2 = mathUtils::multiplication(h1Gate.getInitialEdge(), ket0.getInitialEdge());
    printMemoryUsage();
    printMemoryUsageOnMac();
    return 0;
}