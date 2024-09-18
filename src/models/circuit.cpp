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
                    particalCX0 = mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge());
                    particalCX1 = mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge());
                } else {
                    particalCX0 = mathUtils::kroneckerProduct(particalCX0, mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge()));
                    particalCX1 = mathUtils::kroneckerProduct(particalCX1, mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge()));
                }
            } else if (index == targetIndex) {
                if (index == minIndex) {
                    particalCX1 = gate::X().getInitialEdge();
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
        //     QMDDEdge partialCX0 = mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge());
        //     QMDDEdge partialCX1 = mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge());
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
        //     partialCX0 = mathUtils::kroneckerProduct(partialCX0, mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge()));
        //     partialCX1 = mathUtils::kroneckerProduct(partialCX1, mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge()));
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
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        QMDDEdge particalVarCX0 = gate::I().getInitialEdge();
        QMDDEdge particalVarCX1 = gate::I().getInitialEdge();
        vector<QMDDEdge> edges(targetIndex, gate::I().getInitialEdge());
        for (int index = minIndex; index <= targetIndex; index++){
            if (index == controlIndex) {
                if (index == minIndex) {
                    particalVarCX0 = mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge());
                    particalVarCX1 = mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge());
                } else {
                    particalVarCX0 = mathUtils::kroneckerProduct(particalVarCX0, mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge()));
                    particalVarCX1 = mathUtils::kroneckerProduct(particalVarCX1, mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge()));
                }
            } else if (index == targetIndex) {
                if (index == minIndex) {
                    particalVarCX1 = gate::X().getInitialEdge();
                } else {
                    particalVarCX0 = mathUtils::kroneckerProduct(particalVarCX0, gate::I().getInitialEdge());
                    particalVarCX1 = mathUtils::kroneckerProduct(particalVarCX1, gate::X().getInitialEdge());
                }
            } else {
                particalVarCX0 = mathUtils::kroneckerProduct(particalVarCX0, gate::I().getInitialEdge());
                particalVarCX1 = mathUtils::kroneckerProduct(particalVarCX1, gate::I().getInitialEdge());
            }
        }
        QMDDEdge customCX = mathUtils::addition(particalVarCX0, particalVarCX1);
        edges.push_back(customCX);
        edges.insert(edges.end(), numQubits - maxIndex - 1, gate::I().getInitialEdge());
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
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        QMDDEdge particalCZ0 = gate::I().getInitialEdge();
        QMDDEdge particalCZ1 = gate::I().getInitialEdge();
        vector<QMDDEdge> edges(targetIndex, gate::I().getInitialEdge());
        for (int index = minIndex; index <= targetIndex; index++){
            if (index == controlIndex) {
                if (index == minIndex) {
                    particalCZ0 = mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge());
                    particalCZ1 = mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge());
                } else {
                    particalCZ0 = mathUtils::kroneckerProduct(particalCZ0, mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge()));
                    particalCZ1 = mathUtils::kroneckerProduct(particalCZ1, mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge()));
                }
            } else if (index == targetIndex) {
                if (index == minIndex) {
                    particalCZ1 = gate::Z().getInitialEdge();
                } else {
                    particalCZ0 = mathUtils::kroneckerProduct(particalCZ0, gate::I().getInitialEdge());
                    particalCZ1 = mathUtils::kroneckerProduct(particalCZ1, gate::Z().getInitialEdge());
                }
            } else {
                particalCZ0 = mathUtils::kroneckerProduct(particalCZ0, gate::I().getInitialEdge());
                particalCZ1 = mathUtils::kroneckerProduct(particalCZ1, gate::I().getInitialEdge());
            }
        }
        QMDDEdge customCX = mathUtils::addition(particalCZ0, particalCZ1);
        edges.push_back(customCX);
        edges.insert(edges.end(), numQubits - maxIndex - 1, gate::I().getInitialEdge());
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
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        QMDDEdge particalCP0 = gate::I().getInitialEdge();
        QMDDEdge particalCP1 = gate::I().getInitialEdge();
        vector<QMDDEdge> edges(targetIndex, gate::I().getInitialEdge());
        for (int index = minIndex; index <= targetIndex; index++){
            if (index == controlIndex) {
                if (index == minIndex) {
                    particalCP0 = mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge());
                    particalCP1 = mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge());
                } else {
                    particalCP0 = mathUtils::kroneckerProduct(particalCP0, mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge()));
                    particalCP1 = mathUtils::kroneckerProduct(particalCP1, mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge()));
                }
            } else if (index == targetIndex) {
                if (index == minIndex) {
                    particalCP1 = gate::P(phi).getInitialEdge();
                } else {
                    particalCP0 = mathUtils::kroneckerProduct(particalCP0, gate::I().getInitialEdge());
                    particalCP1 = mathUtils::kroneckerProduct(particalCP1, gate::P(phi).getInitialEdge());
                }
            } else {
                particalCP0 = mathUtils::kroneckerProduct(particalCP0, gate::I().getInitialEdge());
                particalCP1 = mathUtils::kroneckerProduct(particalCP1, gate::I().getInitialEdge());
            }
        }
        QMDDEdge customCX = mathUtils::addition(particalCP0, particalCP1);
        edges.push_back(customCX);
        edges.insert(edges.end(), numQubits - maxIndex - 1, gate::I().getInitialEdge());
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
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        QMDDEdge particalCS0 = gate::I().getInitialEdge();
        QMDDEdge particalCS1 = gate::I().getInitialEdge();
        vector<QMDDEdge> edges(targetIndex, gate::I().getInitialEdge());
        for (int index = minIndex; index <= targetIndex; index++){
            if (index == controlIndex) {
                if (index == minIndex) {
                    particalCS0 = mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge());
                    particalCS1 = mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge());
                } else {
                    particalCS0 = mathUtils::kroneckerProduct(particalCS0, mathUtils::multiplication(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge()));
                    particalCS1 = mathUtils::kroneckerProduct(particalCS1, mathUtils::multiplication(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge()));
                }
            } else if (index == targetIndex) {
                if (index == minIndex) {
                    particalCS1 = gate::S().getInitialEdge();
                } else {
                    particalCS0 = mathUtils::kroneckerProduct(particalCS0, gate::I().getInitialEdge());
                    particalCS1 = mathUtils::kroneckerProduct(particalCS1, gate::S().getInitialEdge());
                }
            } else {
                particalCS0 = mathUtils::kroneckerProduct(particalCS0, gate::I().getInitialEdge());
                particalCS1 = mathUtils::kroneckerProduct(particalCS1, gate::I().getInitialEdge());
            }
        }
        QMDDEdge customCX = mathUtils::addition(particalCS0, particalCS1);
        edges.push_back(customCX);
        edges.insert(edges.end(), numQubits - maxIndex - 1, gate::I().getInitialEdge());
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

void QuantumCircuit::addGate(int qubitIndex, const QMDDGate& gate) {
    if (numQubits == 1) {
        gateQueue.push(gate);
    } else {
        vector<QMDDEdge> edges(qubitIndex, gate::I().getInitialEdge());
        edges.push_back(gate.getInitialEdge());
        edges.insert(edges.end(), numQubits - qubitIndex - 1, gate::I().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kroneckerProduct);
        gateQueue.push(result);
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