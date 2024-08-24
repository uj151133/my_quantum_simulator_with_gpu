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