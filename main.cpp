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
#include "src/common/mathUtils.hpp"
#include "src/common/calculation.hpp"

using namespace GiNaC;
using namespace std;



void printMemoryUsage() {
    pid_t pid = getpid();
    string command = "ps -o rss= -p " + to_string(pid);

    // Create a pipe to capture the output of the command
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        cerr << "Failed to run command.\n";
        return;
    }

    // Read the output of the command
    char buffer[128];
    string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

    // Close the pipe and check for errors
    pclose(pipe);

    // Remove any trailing whitespace from the result
    result.erase(result.find_last_not_of(" \n\r\t") + 1);

    cout << "Memory usage: " << result << " KB" << endl;
}

void printMemoryUsageOnMac() {
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
        cerr << "Error getting memory info\n";
        return;
    }

    cout << "Memory usage on mac enviroment: " << info.resident_size / 1024 << " KB\n";
}

bool isExecuteGui() {
    YAML::Node config = YAML::LoadFile("config.yaml");
    bool guiEnabled = config["gui"]["enabled"].as<bool>();

    return guiEnabled;
}



int main() {
    // printMemoryUsage();
    // printMemoryUsageOnMac();

    bool isGuiEnabled = isExecuteGui();

    if (isGuiEnabled) {
        cout << "GUI is enabled." << endl;
    } else {
        cout << "GUI is disabled." << endl;
    }

    UniqueTable& uniqueTable = UniqueTable::getInstance();

    QMDDGate zeroGate = gate::O();
    // QMDDGate iGate = gate::I();
    // QMDDGate phGate = gate::Ph(0.5);
    // QMDDGate xGate = gate::X();
    // QMDDGate hGate = gate::H();
    // QMDDGate sGate = gate::S();
    QMDDGate cx1Gate = gate::CX1();
    QMDDGate cx2Gate = gate::CX2();
    // cout << "zeroGate:" << zeroGate.getInitialEdge() << endl;
    // cout << "zeroGate:" << zeroGate << endl;
    // cout << "igate:" << iGate.getDepth() << endl;
    // cout << "phgate:" << phGate.getInitialEdge() << endl;
    cout << "cx1gate:" << cx2Gate.getInitialEdge() << endl;
    // cout << "cx2gate:" << cx2Gate.getDepth() << endl;
    // cout << "igate:" << gate::I().getInitialEdge() << endl;
    // cout << "x1gate:" << gate::X().getInitialEdge() << endl;
    // QMDDGate h2Gate = gate::H();
    // cout << "h2gate:" << h2Gate.getInitialEdge() << endl;

    // QMDDGate xGate = gate::X();
    // cout << "xgate:" << xGate.getInitialEdge() << endl;
    // QMDDState ket0 = state::KET_0();
    // auto result1 = mathUtils::addition(cx1Gate.getInitialEdge(), cx2Gate.getInitialEdge());
    // cout << "result1:" << result1 << endl;
    // auto result2 = mathUtils::addition(xGate.getInitialEdge(), iGate.getInitialEdge());
    // cout << "result2:" << result2 << endl;

    uniqueTable.printAllEntries();


    // QMDDGate cx1 = gate::CX1();
    // QMDDGate cx2 = gate::CX2();
    // auto result2 = mathUtils::addition(cx1.getInitialEdge(), cx2.getInitialEdge());
    // printMemoryUsage();
    // printMemoryUsageOnMac();
    return 0;
}