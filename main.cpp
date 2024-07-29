#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <mach/mach.h>
#include <ginac/ginac.h>
#include "src/models/bit.hpp"
#include "src/models/gate.hpp"
#include "src/models/uniquetable.hpp"
#include "src/models/qmdd.hpp"
#include "src/common/calculation.hpp"
#include "src/common/mathUtils.hpp"

using namespace GiNaC;

void printMemoryUsage() {
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
        std::cerr << "Error getting memory info\n";
        return;
    }

    std::cout << "Memory usage: " << info.resident_size / 1024 << " KB\n";
}

int main() {
    printMemoryUsage();
    QMDDGate hGate = gate::H();
    cout << "hgate:" << hGate.getInitialEdge() << endl;
    QMDDGate xGate = gate::X();
    cout << "xgate:" << xGate.getInitialEdge() << endl;
    QMDDState ket0 = state::KET_0();
    auto result1 = mathUtils::addition(hGate.getInitialEdge(), ket0.getInitialEdge());
    cout << "result:" << result1 << endl;

    auto result2 = mathUtils::multiplication(hGate.getInitialEdge(), ket0.getInitialEdge());
    printMemoryUsage();
    return 0;
}