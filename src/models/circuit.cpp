#include "circuit.hpp"

void QuantumCircuit::addI(int qubitIndex) {
    vector<QMDDGate> gates(numQubits, gate::I());
    QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
    gateQueue.push(result);
    return;
}

void QuantumCircuit::addPh(int qubitIndex, double delta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Ph(delta));
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::Ph(delta));
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addX(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::X());
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::X());
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addY(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::Y());
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::Y());
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addZ(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::Z());
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::Z());
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addS(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::S());
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::S());
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addV(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::V());
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::V());
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addH(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::H());
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::H());
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCX(int controlIndex, int targetIndex) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CX gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        gateQueue.push(gate::CX1());
    }else if(numQubits == 2 && controlIndex == 1 && targetIndex == 0) {
        gateQueue.push(gate::CX2());
    }else if(controlIndex < targetIndex) {
        vector<QMDDGate> gates(controlIndex, gate::I());
        QMDDEdge partialCX0 = mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge());
        QMDDEdge partialCX1 = mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge());
        for (int i = 0; i < targetIndex - controlIndex - 1; i++) {
            partialCX0 = mathUtils::kroneckerProduct(partialCX0, gate::I().getInitialEdge());
            partialCX1 = mathUtils::kroneckerProduct(partialCX1, gate::I().getInitialEdge());
        }
        partialCX0 = mathUtils::kroneckerProduct(partialCX0, gate::I().getInitialEdge());
        partialCX1 = mathUtils::kroneckerProduct(partialCX1, gate::X().getInitialEdge());
        QMDDEdge customCX = mathUtils::addition(partialCX0, partialCX1);
        gates.push_back(customCX);
        gates.insert(gates.end(), numQubits - targetIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }else {
        vector<QMDDGate> gates(targetIndex, gate::I());
        QMDDEdge partialCX0 = gate::I().getInitialEdge();
        QMDDEdge partialCX1 = gate::X().getInitialEdge();
        for (int i = 0; i < controlIndex - targetIndex - 1; i++) {
            partialCX0 = mathUtils::kroneckerProduct(partialCX0, gate::I().getInitialEdge());
            partialCX1 = mathUtils::kroneckerProduct(partialCX1, gate::I().getInitialEdge());
        }
        partialCX0 = mathUtils::kroneckerProduct(partialCX0, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
        partialCX1 = mathUtils::kroneckerProduct(partialCX1, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
        QMDDEdge customCX = mathUtils::addition(partialCX0, partialCX1);
        gates.push_back(customCX);
        gates.insert(gates.end(), numQubits - controlIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addVarCX(int controlIndex, int targetIndex) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add var CX gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        gateQueue.push(gate::varCX());
    }else if(controlIndex < targetIndex) {
        vector<QMDDGate> gates(controlIndex, gate::I());
        QMDDEdge partialVarCX0 = mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge());
        QMDDEdge partialVarCX1 = mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge());
        for (int i = 0; i < targetIndex - controlIndex - 1; i++) {
            partialVarCX0 = mathUtils::kroneckerProduct(partialVarCX0, gate::I().getInitialEdge());
            partialVarCX1 = mathUtils::kroneckerProduct(partialVarCX1, gate::I().getInitialEdge());
        }
        partialVarCX0 = mathUtils::kroneckerProduct(partialVarCX0, gate::I().getInitialEdge());
        partialVarCX1 = mathUtils::kroneckerProduct(partialVarCX1, gate::X().getInitialEdge());
        QMDDEdge customVarCX = mathUtils::addition(partialVarCX0, partialVarCX1);
        gates.push_back(customVarCX);
        gates.insert(gates.end(), numQubits - targetIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }else {
        vector<QMDDGate> gates(targetIndex, gate::I());
        QMDDEdge partialVarCX0 = gate::I().getInitialEdge();
        QMDDEdge partialVarCX1 = gate::X().getInitialEdge();
        for (int i = 0; i < controlIndex - targetIndex - 1; i++) {
            partialVarCX0 = mathUtils::kroneckerProduct(partialVarCX0, gate::I().getInitialEdge());
            partialVarCX1 = mathUtils::kroneckerProduct(partialVarCX1, gate::I().getInitialEdge());
        }
        partialVarCX0 = mathUtils::kroneckerProduct(partialVarCX0, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
        partialVarCX1 = mathUtils::kroneckerProduct(partialVarCX1, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
        QMDDEdge customVarCX = mathUtils::addition(partialVarCX0, partialVarCX1);
        gates.push_back(customVarCX);
        gates.insert(gates.end(), numQubits - controlIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCZ(int controlIndex, int targetIndex) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CZ gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        gateQueue.push(gate::CZ());
    }else if(controlIndex < targetIndex) {
        vector<QMDDGate> gates(controlIndex, gate::I());
        QMDDEdge partialCZ0 = mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge());
        QMDDEdge partialCZ1 = mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge());
        for (int i = 0; i < targetIndex - controlIndex - 1; i++) {
            partialCZ0 = mathUtils::kroneckerProduct(partialCZ0, gate::I().getInitialEdge());
            partialCZ1 = mathUtils::kroneckerProduct(partialCZ1, gate::I().getInitialEdge());
        }
        partialCZ0 = mathUtils::kroneckerProduct(partialCZ0, gate::I().getInitialEdge());
        partialCZ1 = mathUtils::kroneckerProduct(partialCZ1, gate::Z().getInitialEdge());
        QMDDEdge customCZ = mathUtils::addition(partialCZ0, partialCZ1);
        gates.push_back(customCZ);
        gates.insert(gates.end(), numQubits - targetIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }else {
        vector<QMDDGate> gates(targetIndex, gate::I());
        QMDDEdge partialCZ0 = gate::I().getInitialEdge();
        QMDDEdge partialCZ1 = gate::Z().getInitialEdge();
        for (int i = 0; i < controlIndex - targetIndex - 1; i++) {
            partialCZ0 = mathUtils::kroneckerProduct(partialCZ0, gate::I().getInitialEdge());
            partialCZ1 = mathUtils::kroneckerProduct(partialCZ1, gate::I().getInitialEdge());
        }
        partialCZ0 = mathUtils::kroneckerProduct(partialCZ0, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
        partialCZ1 = mathUtils::kroneckerProduct(partialCZ1, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
        QMDDEdge customCZ = mathUtils::addition(partialCZ0, partialCZ1);
        gates.push_back(customCZ);
        gates.insert(gates.end(), numQubits - controlIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}


void QuantumCircuit::addP(int qubitIndex, double phi) {
    if (numQubits == 1) {
        gateQueue.push(gate::P(phi));
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::P(phi));
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addT(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::T());
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::T());
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCP(int controlIndex, int targetIndex, double phi) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CP gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        gateQueue.push(gate::CP(phi));
    }else if(controlIndex < targetIndex) {
        vector<QMDDGate> gates(controlIndex, gate::I());
        QMDDEdge partialCP0 = mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge());
        QMDDEdge partialCP1 = mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge());
        for (int i = 0; i < targetIndex - controlIndex - 1; i++) {
            partialCP0 = mathUtils::kroneckerProduct(partialCP0, gate::I().getInitialEdge());
            partialCP1 = mathUtils::kroneckerProduct(partialCP1, gate::I().getInitialEdge());
        }
        partialCP0 = mathUtils::kroneckerProduct(partialCP0, gate::I().getInitialEdge());
        partialCP1 = mathUtils::kroneckerProduct(partialCP1, gate::P(phi).getInitialEdge());
        QMDDEdge customCP = mathUtils::addition(partialCP0, partialCP1);
        gates.push_back(customCP);
        gates.insert(gates.end(), numQubits - targetIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }else {
        vector<QMDDGate> gates(targetIndex, gate::I());
        QMDDEdge partialCP0 = gate::I().getInitialEdge();
        QMDDEdge partialCP1 = gate::P(phi).getInitialEdge();
        for (int i = 0; i < controlIndex - targetIndex - 1; i++) {
            partialCP0 = mathUtils::kroneckerProduct(partialCP0, gate::I().getInitialEdge());
            partialCP1 = mathUtils::kroneckerProduct(partialCP1, gate::I().getInitialEdge());
        }
        partialCP0 = mathUtils::kroneckerProduct(partialCP0, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
        partialCP1 = mathUtils::kroneckerProduct(partialCP1, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
        QMDDEdge customCP = mathUtils::addition(partialCP0, partialCP1);
        gates.push_back(customCP);
        gates.insert(gates.end(), numQubits - controlIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCS(int controlIndex, int targetIndex) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CS gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        gateQueue.push(gate::CS());
    }else if(controlIndex < targetIndex) {
        vector<QMDDGate> gates(controlIndex, gate::I());
        QMDDEdge partialCS0 = mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge());
        QMDDEdge partialCS1 = mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge());
        for (int i = 0; i < targetIndex - controlIndex - 1; i++) {
            partialCS0 = mathUtils::kroneckerProduct(partialCS0, gate::I().getInitialEdge());
            partialCS1 = mathUtils::kroneckerProduct(partialCS1, gate::I().getInitialEdge());
        }
        partialCS0 = mathUtils::kroneckerProduct(partialCS0, gate::I().getInitialEdge());
        partialCS1 = mathUtils::kroneckerProduct(partialCS1, gate::S().getInitialEdge());
        QMDDEdge customCS = mathUtils::addition(partialCS0, partialCS1);
        gates.push_back(customCS);
        gates.insert(gates.end(), numQubits - targetIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }else {
        vector<QMDDGate> gates(targetIndex, gate::I());
        QMDDEdge partialCS0 = gate::I().getInitialEdge();
        QMDDEdge partialCS1 = gate::S().getInitialEdge();
        for (int i = 0; i < controlIndex - targetIndex - 1; i++) {
            partialCS0 = mathUtils::kroneckerProduct(partialCS0, gate::I().getInitialEdge());
            partialCS1 = mathUtils::kroneckerProduct(partialCS1, gate::I().getInitialEdge());
        }
        partialCS0 = mathUtils::kroneckerProduct(partialCS0, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
        partialCS1 = mathUtils::kroneckerProduct(partialCS1, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
        QMDDEdge customCS = mathUtils::addition(partialCS0, partialCS1);
        gates.push_back(customCS);
        gates.insert(gates.end(), numQubits - controlIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}


void QuantumCircuit::addRx(int qubitIndex, double theta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Rx(theta));
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::Rx(theta));
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addRy(int qubitIndex, double theta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Ry(theta));
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::Ry(theta));
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addRz(int qubitIndex, double theta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Rz(theta));
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::Rz(theta));
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addU(int qubitIndex, double theta, double phi, double lambda) {
    if (numQubits == 1) {
        gateQueue.push(gate::U(theta, phi, lambda));
    } else {
        vector<QMDDGate> gates(qubitIndex, gate::I());
        gates.push_back(gate::U(theta, phi, lambda));
        gates.insert(gates.end(), numQubits - qubitIndex - 1, gate::I());
        QMDDGate result = accumulate(gates.begin() + 1, gates.end(), gates[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}