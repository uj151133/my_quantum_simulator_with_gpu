#include "circuit.hpp"

QuantumCircuit::QuantumCircuit(int numQubits, QMDDState initialState) : numQubits(numQubits), initialState(initialState), finalState(initialState) {
    if (numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }
}

void QuantumCircuit::addI(int qubitIndex) {
    vector<QMDDEdge> edges(numQubits, gate::I().getInitialEdge());
    QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
    gateQueue.push(result);
    return;
}

void QuantumCircuit::addPh(int qubitIndex, double delta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Ph(delta));
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::Ph(delta).getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addX(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::X());
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::X().getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addY(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::Y());
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::Y().getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addZ(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::Z());
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::Z().getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addS(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::S());
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::S().getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addV(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::V());
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::V().getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addH(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::H());
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::H().getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
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
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        QMDDEdge particalCX0 = gate::I().getInitialEdge();
        QMDDEdge particalCX1 = gate::I().getInitialEdge();
        vector<QMDDEdge> edges(targetIndex, gate::I().getInitialEdge());
        for (int index = minIndex; index <= targetIndex; index++){
            if (index == controlIndex) {
                if (index == minIndex) {
                    particalCX0 = mathUtils::multiplication(particalCX0, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
                    particalCX1 = mathUtils::multiplication(particalCX1, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
                } else {
                    particalCX0 = mathUtils::kroneckerProduct(particalCX0, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
                    particalCX1 = mathUtils::kroneckerProduct(particalCX1, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
                }
            } else if (index == targetIndex) {
                if (index == minIndex) {
                    particalCX0 = mathUtils::multiplication(particalCX0, gate::I().getInitialEdge());
                    particalCX1 = mathUtils::multiplication(particalCX1, gate::X().getInitialEdge());
                } else {
                    particalCX0 = mathUtils::kroneckerProduct(particalCX0, gate::I().getInitialEdge());
                    particalCX1 = mathUtils::kroneckerProduct(particalCX1, gate::X().getInitialEdge());
                }
            } else {
                particalCX0 = mathUtils::kroneckerProduct(particalCX0, gate::I().getInitialEdge());
                particalCX1 = mathUtils::kroneckerProduct(particalCX1, gate::I().getInitialEdge());
            }
        }
        // if(controlIndex < targetIndex) {
        //     QMDDEdge partialCX0 = mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge());
        //     QMDDEdge partialCX1 = mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge());
        //     for (int i = 0; i < targetIndex - controlIndex - 1; i++) {
        //         partialCX0 = mathUtils::kroneckerProduct(partialCX0, gate::I().getInitialEdge());
        //         partialCX1 = mathUtils::kroneckerProduct(partialCX1, gate::I().getInitialEdge());
        //     }
        //     partialCX0 = mathUtils::kroneckerProduct(partialCX0, gate::I().getInitialEdge());
        //     partialCX1 = mathUtils::kroneckerProduct(partialCX1, gate::X().getInitialEdge());
        // }else {
        //     QMDDEdge partialCX0 = gate::I().getInitialEdge();
        //     QMDDEdge partialCX1 = gate::X().getInitialEdge();
        //     for (int i = 0; i < controlIndex - targetIndex - 1; i++) {
        //         partialCX0 = mathUtils::kroneckerProduct(partialCX0, gate::I().getInitialEdge());
        //         partialCX1 = mathUtils::kroneckerProduct(partialCX1, gate::I().getInitialEdge());
        //     }
        //     partialCX0 = mathUtils::kroneckerProduct(partialCX0, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
        //     partialCX1 = mathUtils::kroneckerProduct(partialCX1, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
        // }
        QMDDEdge customCX = mathUtils::addition(particalCX0, particalCX1);
        edges.push_back(customCX);
        edges.insert(edges.end(), numQubits - maxIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
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
        vector<QMDDEdge> edges(controlIndex, gate::I().getInitialEdge());
        QMDDEdge partialVarCX0 = mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge());
        QMDDEdge partialVarCX1 = mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge());
        for (int i = 0; i < targetIndex - controlIndex - 1; i++) {
            partialVarCX0 = mathUtils::kroneckerProduct(partialVarCX0, gate::I().getInitialEdge());
            partialVarCX1 = mathUtils::kroneckerProduct(partialVarCX1, gate::I().getInitialEdge());
        }
        partialVarCX0 = mathUtils::kroneckerProduct(partialVarCX0, gate::I().getInitialEdge());
        partialVarCX1 = mathUtils::kroneckerProduct(partialVarCX1, gate::X().getInitialEdge());
        QMDDEdge customVarCX = mathUtils::addition(partialVarCX0, partialVarCX1);
        edges.push_back(customVarCX);
        edges.insert(edges.end(), numQubits - targetIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }else {
        vector<QMDDEdge> edges(targetIndex, gate::I().getInitialEdge());
        QMDDEdge partialVarCX0 = gate::I().getInitialEdge();
        QMDDEdge partialVarCX1 = gate::X().getInitialEdge();
        for (int i = 0; i < controlIndex - targetIndex - 1; i++) {
            partialVarCX0 = mathUtils::kroneckerProduct(partialVarCX0, gate::I().getInitialEdge());
            partialVarCX1 = mathUtils::kroneckerProduct(partialVarCX1, gate::I().getInitialEdge());
        }
        partialVarCX0 = mathUtils::kroneckerProduct(partialVarCX0, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
        partialVarCX1 = mathUtils::kroneckerProduct(partialVarCX1, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
        QMDDEdge customVarCX = mathUtils::addition(partialVarCX0, partialVarCX1);
        edges.push_back(customVarCX);
        edges.insert(edges.end(), numQubits - controlIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
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
        vector<QMDDEdge> edges(controlIndex, gate::I().getInitialEdge());
        QMDDEdge partialCZ0 = mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge());
        QMDDEdge partialCZ1 = mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge());
        for (int i = 0; i < targetIndex - controlIndex - 1; i++) {
            partialCZ0 = mathUtils::kroneckerProduct(partialCZ0, gate::I().getInitialEdge());
            partialCZ1 = mathUtils::kroneckerProduct(partialCZ1, gate::I().getInitialEdge());
        }
        partialCZ0 = mathUtils::kroneckerProduct(partialCZ0, gate::I().getInitialEdge());
        partialCZ1 = mathUtils::kroneckerProduct(partialCZ1, gate::Z().getInitialEdge());
        QMDDEdge customCZ = mathUtils::addition(partialCZ0, partialCZ1);
        edges.push_back(customCZ);
        edges.insert(edges.end(), numQubits - targetIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }else {
        vector<QMDDEdge> edges(targetIndex, gate::I().getInitialEdge());
        QMDDEdge partialCZ0 = gate::I().getInitialEdge();
        QMDDEdge partialCZ1 = gate::Z().getInitialEdge();
        for (int i = 0; i < controlIndex - targetIndex - 1; i++) {
            partialCZ0 = mathUtils::kroneckerProduct(partialCZ0, gate::I().getInitialEdge());
            partialCZ1 = mathUtils::kroneckerProduct(partialCZ1, gate::I().getInitialEdge());
        }
        partialCZ0 = mathUtils::kroneckerProduct(partialCZ0, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
        partialCZ1 = mathUtils::kroneckerProduct(partialCZ1, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
        QMDDEdge customCZ = mathUtils::addition(partialCZ0, partialCZ1);
        edges.push_back(customCZ);
        edges.insert(edges.end(), numQubits - controlIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}


void QuantumCircuit::addP(int qubitIndex, double phi) {
    if (numQubits == 1) {
        gateQueue.push(gate::P(phi));
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::P(phi).getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addT(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::T());
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::T().getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
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
        vector<QMDDEdge> edges(controlIndex, gate::I().getInitialEdge());
        QMDDEdge partialCP0 = mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge());
        QMDDEdge partialCP1 = mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge());
        for (int i = 0; i < targetIndex - controlIndex - 1; i++) {
            partialCP0 = mathUtils::kroneckerProduct(partialCP0, gate::I().getInitialEdge());
            partialCP1 = mathUtils::kroneckerProduct(partialCP1, gate::I().getInitialEdge());
        }
        partialCP0 = mathUtils::kroneckerProduct(partialCP0, gate::I().getInitialEdge());
        partialCP1 = mathUtils::kroneckerProduct(partialCP1, gate::P(phi).getInitialEdge());
        QMDDEdge customCP = mathUtils::addition(partialCP0, partialCP1);
        edges.push_back(customCP);
        edges.insert(edges.end(), numQubits - targetIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }else {
        vector<QMDDEdge> edges(targetIndex, gate::I().getInitialEdge());
        QMDDEdge partialCP0 = gate::I().getInitialEdge();
        QMDDEdge partialCP1 = gate::P(phi).getInitialEdge();
        for (int i = 0; i < controlIndex - targetIndex - 1; i++) {
            partialCP0 = mathUtils::kroneckerProduct(partialCP0, gate::I().getInitialEdge());
            partialCP1 = mathUtils::kroneckerProduct(partialCP1, gate::I().getInitialEdge());
        }
        partialCP0 = mathUtils::kroneckerProduct(partialCP0, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
        partialCP1 = mathUtils::kroneckerProduct(partialCP1, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
        QMDDEdge customCP = mathUtils::addition(partialCP0, partialCP1);
        edges.push_back(customCP);
        edges.insert(edges.end(), numQubits - controlIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
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
        vector<QMDDEdge> edges(controlIndex, gate::I().getInitialEdge());
        QMDDEdge partialCS0 = mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge());
        QMDDEdge partialCS1 = mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge());
        for (int i = 0; i < targetIndex - controlIndex - 1; i++) {
            partialCS0 = mathUtils::kroneckerProduct(partialCS0, gate::I().getInitialEdge());
            partialCS1 = mathUtils::kroneckerProduct(partialCS1, gate::I().getInitialEdge());
        }
        partialCS0 = mathUtils::kroneckerProduct(partialCS0, gate::I().getInitialEdge());
        partialCS1 = mathUtils::kroneckerProduct(partialCS1, gate::S().getInitialEdge());
        QMDDEdge customCS = mathUtils::addition(partialCS0, partialCS1);
        edges.push_back(customCS);
        edges.insert(edges.end(), numQubits - targetIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }else {
        vector<QMDDEdge> edges(targetIndex, gate::I().getInitialEdge());
        QMDDEdge partialCS0 = gate::I().getInitialEdge();
        QMDDEdge partialCS1 = gate::S().getInitialEdge();
        for (int i = 0; i < controlIndex - targetIndex - 1; i++) {
            partialCS0 = mathUtils::kroneckerProduct(partialCS0, gate::I().getInitialEdge());
            partialCS1 = mathUtils::kroneckerProduct(partialCS1, gate::I().getInitialEdge());
        }
        partialCS0 = mathUtils::kroneckerProduct(partialCS0, mathUtils::multiplication(state::KET_0().getInitialEdge(), state::BRA_0().getInitialEdge()));
        partialCS1 = mathUtils::kroneckerProduct(partialCS1, mathUtils::multiplication(state::KET_1().getInitialEdge(), state::BRA_1().getInitialEdge()));
        QMDDEdge customCS = mathUtils::addition(partialCS0, partialCS1);
        edges.push_back(customCS);
        edges.insert(edges.end(), numQubits - controlIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}


void QuantumCircuit::addRx(int qubitIndex, double theta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Rx(theta));
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::Rx(theta).getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addRy(int qubitIndex, double theta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Ry(theta));
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::Ry(theta).getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addRz(int qubitIndex, double theta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Rz(theta));
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::Rz(theta).getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addU(int qubitIndex, double theta, double phi, double lambda) {
    if (numQubits == 1) {
        gateQueue.push(gate::U(theta, phi, lambda));
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate::U(theta, phi, lambda).getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addToff(const vector<int>& controlIndexes, int targetIndex) {
    if (controlIndexes.size() == 0) {
        throw invalid_argument("Control indexes must not be empty.");
    }else if (numQubits < controlIndexes.size() + 1) {
        throw invalid_argument("Number of control indexes must be at most number of qubits - 1.");
    }else if (controlIndexes.size() == 1) {
        addCX(controlIndexes[0], targetIndex);
    }else {
        int minIndex = min(*min_element(controlIndexes.begin(), controlIndexes.end()), targetIndex);
        int maxIndex = max(*max_element(controlIndexes.begin(), controlIndexes.end()), targetIndex);
        vector<QMDDEdge> edges(minIndex, gate::I().getInitialEdge());
        vector<QMDDEdge> particalToff[controlIndexes.size() + 1];
        for (int i = minIndex; i <= maxIndex; i++) {
            if (find(controlIndexes.begin(), controlIndexes.end(), i) != controlIndexes.end()) {
                edges.push_back(gate::I().getInitialEdge());
            }else {
                edges.push_back(gate::Z().getInitialEdge());
            }
        }


    }
    return;
}

void QuantumCircuit::execute() {
    QMDDState currentState = initialState;
    while (!gateQueue.empty()) {
        QMDDGate currentGate = gateQueue.front();
        cout << "Current gate: " << currentGate << endl;
        cout << "Current state: " << currentState << endl;
        UniqueTable& uniqueTable = UniqueTable::getInstance();
        uniqueTable.printAllEntries();
        gateQueue.pop();
        currentState = mathUtils::multiplication(currentGate.getInitialEdge(), currentState.getInitialEdge());
    }
    finalState = currentState;
    cout << "Final state: " << finalState << endl;
    return;
}