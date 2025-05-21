#include "circuit.hpp"


QuantumCircuit::QuantumCircuit(int numQubits, QMDDState initialState) : numQubits(numQubits), finalState(initialState) {
    call_once(initExtendedEdgeFlag, initExtendedEdge);
    if (numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }
}

QuantumCircuit::QuantumCircuit(int numQubits) : numQubits(numQubits), finalState(state::Ket0()) {

    call_once(initExtendedEdgeFlag, initExtendedEdge);
    if (numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }

    for (int i = 1; i < numQubits; i++) {
        this->finalState = mathUtils::kron(this->finalState.getInitialEdge(), state::Ket0().getInitialEdge());
    }
}

queue<QMDDGate> QuantumCircuit::getGateQueue() const {
    return this->gateQueue;
}

QMDDState QuantumCircuit::getFinalState() const {
    return this->finalState;
}

void QuantumCircuit::addI(int qubitIndex) {
    return;
}

void QuantumCircuit::addPh(int qubitIndex, double delta) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::Ph(delta));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::Ph(delta).getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronForDiagonal);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addX(vector<int> qubitIndices) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::X());
    } else {
        sort(qubitIndices.begin(), qubitIndices.end());
        vector<QMDDEdge> edges;
        for (int qubitIndex : qubitIndices) {
            edges.insert(edges.end(), qubitIndex - edges.size(), identityEdge);
            edges.push_back(gate::X().getInitialEdge());
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addAllX() {
    vector<QMDDEdge> edges(this->numQubits, gate::X().getInitialEdge());
    QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
    this->gateQueue.push(result);
    return;
}

void QuantumCircuit::addY(vector<int> qubitIndices) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::Y());
    } else {
        sort(qubitIndices.begin(), qubitIndices.end());
        vector<QMDDEdge> edges;
        for (int qubitIndex : qubitIndices) {
            edges.insert(edges.end(), qubitIndex - edges.size(), identityEdge);
            edges.push_back(gate::Y().getInitialEdge());
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addZ(vector<int> qubitIndices) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::Z());
    } else {
        sort(qubitIndices.begin(), qubitIndices.end());
        vector<QMDDEdge> edges;
        for (int qubitIndex : qubitIndices) {
            edges.insert(edges.end(), qubitIndex - edges.size(), identityEdge);
            edges.push_back(gate::Z().getInitialEdge());
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronForDiagonal);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addS(vector<int> qubitIndices) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::S());
    } else {
        sort(qubitIndices.begin(), qubitIndices.end());
        vector<QMDDEdge> edges;
        for (int qubitIndex : qubitIndices) {
            edges.insert(edges.end(), qubitIndex - edges.size(), identityEdge);
            edges.push_back(gate::S().getInitialEdge());
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronForDiagonal);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addSdg(vector<int> qubitIndices) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::Sdg());
    } else {
        sort(qubitIndices.begin(), qubitIndices.end());
        vector<QMDDEdge> edges;
        for (int qubitIndex : qubitIndices) {
            edges.insert(edges.end(), qubitIndex - edges.size(), identityEdge);
            edges.push_back(gate::Sdg().getInitialEdge());
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronForDiagonal);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addV(vector<int> qubitIndices) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::V());
    } else {
        sort(qubitIndices.begin(), qubitIndices.end());
        vector<QMDDEdge> edges;
        for (int qubitIndex : qubitIndices) {
            edges.insert(edges.end(), qubitIndex - edges.size(), identityEdge);
            edges.push_back(gate::V().getInitialEdge());
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addH(vector<int> qubitIndices) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::H());
    } else {
        sort(qubitIndices.begin(), qubitIndices.end());
        vector<QMDDEdge> edges;
        for (int qubitIndex : qubitIndices) {
            edges.insert(edges.end(), qubitIndex - edges.size(), identityEdge);
            edges.push_back(gate::H().getInitialEdge());
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addAllH() {
    vector<QMDDEdge> edges(numQubits, gate::H().getInitialEdge());
    QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
    this->gateQueue.push(result);
    return;
}

void QuantumCircuit::addCX(int controlIndex, int targetIndex) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CX gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        this->gateQueue.push(gate::CX1());
    }else if(numQubits == 2 && controlIndex == 1 && targetIndex == 0) {
        this->gateQueue.push(gate::CX2());
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        QMDDEdge customCX;
        vector<QMDDEdge> edges(minIndex, identityEdge);
        if (maxIndex - minIndex == 1) {
            if (minIndex == controlIndex) {
                customCX = gate::CX1().getInitialEdge();
            } else {
                customCX = gate::CX2().getInitialEdge();
            }
        } else {
            array<QMDDEdge, 2> partialCX;
            if (minIndex == controlIndex) {
                partialCX[0] = braketZero;
                partialCX[1] = braketOne;
            } else {
                partialCX[0] = identityEdge;
                partialCX[1] = gate::X().getInitialEdge();
            }

            for (int index = minIndex + 1; index <= maxIndex; index++){
                if (index == controlIndex) {
                    partialCX[0] = mathUtils::kron(partialCX[0], braketZero);
                    partialCX[1] = mathUtils::kron(partialCX[1], braketOne);
                } else if (index == targetIndex) {
                    partialCX[0] = mathUtils::kron(partialCX[0], identityEdge);
                    partialCX[1] = mathUtils::kron(partialCX[1], gate::X().getInitialEdge());
                } else {
                    partialCX[0] = mathUtils::kron(partialCX[0], identityEdge);
                    partialCX[1] = mathUtils::kron(partialCX[1], identityEdge);
                }
            }
            customCX = mathUtils::add(partialCX[0], partialCX[1]);
        }
        edges.push_back(customCX);
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addVarCX(int controlIndex, int targetIndex) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add var CX gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        this->gateQueue.push(gate::varCX());
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        if (targetIndex - controlIndex == 1) {
            edges.push_back(gate::varCX().getInitialEdge());
        }
        else {
            array<QMDDEdge, 2> partialVarCX;
            if (minIndex == controlIndex) {
                partialVarCX[0] = braketOne;
                partialVarCX[1] = braketZero;
            } else {
                partialVarCX[0] = identityEdge;
                partialVarCX[1] = gate::X().getInitialEdge();
            }
            for (int index = minIndex + 1; index <= maxIndex; index++){
                if (index == controlIndex) {
                    partialVarCX[0] = mathUtils::kron(partialVarCX[0], braketOne);
                    partialVarCX[1] = mathUtils::kron(partialVarCX[1], braketZero);
                } else if (index == targetIndex) {
                    partialVarCX[0] = mathUtils::kron(partialVarCX[0], identityEdge);
                    partialVarCX[1] = mathUtils::kron(partialVarCX[1], gate::X().getInitialEdge());
                } else {
                    partialVarCX[0] = mathUtils::kron(partialVarCX[0], identityEdge);
                    partialVarCX[1] = mathUtils::kron(partialVarCX[1], identityEdge);
                }
            }
            QMDDEdge customVarCX = mathUtils::add(partialVarCX[0], partialVarCX[1]);
            edges.push_back(customVarCX);
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCZ(int controlIndex, int targetIndex) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CZ gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        this->gateQueue.push(gate::CZ());
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        if (targetIndex - controlIndex == 1) {
            edges.push_back(gate::CZ().getInitialEdge());
        } else {
            array<QMDDEdge, 2> partialCZ;
            if (minIndex == controlIndex) {
                partialCZ[0] = braketZero;
                partialCZ[1] = braketOne;
            } else {
                partialCZ[0] = identityEdge;
                partialCZ[1] = gate::Z().getInitialEdge();
            }
            for (int index = minIndex + 1; index <= maxIndex; index++){
                if (index == controlIndex) {
                    partialCZ[0] = mathUtils::kron(partialCZ[0], braketZero);
                    partialCZ[1] = mathUtils::kron(partialCZ[1], braketOne);
                } else if (index == targetIndex) {
                    partialCZ[0] = mathUtils::kron(partialCZ[0], identityEdge);
                    partialCZ[1] = mathUtils::kron(partialCZ[1], gate::Z().getInitialEdge());
                } else {
                    partialCZ[0] = mathUtils::kron(partialCZ[0], identityEdge);
                    partialCZ[1] = mathUtils::kron(partialCZ[1], identityEdge);
                }
            }
            QMDDEdge customCZ = mathUtils::add(partialCZ[0], partialCZ[1]);
            edges.push_back(customCZ);
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addSWAP(int qubitIndex1, int qubitIndex2) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add SWAP gate to single qubit circuit.");
    }else if (qubitIndex1 == qubitIndex2) {
        throw invalid_argument("qubitIndexes indices must be different.");
    }else if(numQubits == 2 && ((qubitIndex1 == 0 && qubitIndex2 == 1) || (qubitIndex1 == 1 && qubitIndex2 == 0))) {
        this->gateQueue.push(gate::SWAP());
    }else {
        int minIndex = min(qubitIndex1, qubitIndex2);
        int maxIndex = max(qubitIndex1, qubitIndex2);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        QMDDEdge customSWAP;
        size_t numIndex =  maxIndex - minIndex + 1;
        if (numIndex == 2){
            customSWAP = gate::SWAP().getInitialEdge();
        } else {
            vector<vector<QMDDEdge>> partialPreSWAP(pow(2, numIndex), vector<QMDDEdge>(2, identityEdge));
            vector<QMDDEdge> partialSWAP(pow(2, numIndex));
            for (size_t i = 0; i < partialPreSWAP.size(); i++){
                unsigned long long highest_bit = (i >> (numIndex - 1)) & 1;
                unsigned long long lowest_bit = i & 1;
                int j = (i & ~(1ULL << (numIndex - 1))) | (lowest_bit << (numIndex - 1));
                j = (j & ~1ULL) | highest_bit;
                for (int index = 0; index < numIndex; index++){
                    if (i & 1){
                        if (index == 0){
                            partialPreSWAP[i][0] = state::Ket1().getInitialEdge();
                        }else {
                            partialPreSWAP[i][0] = mathUtils::kron(partialPreSWAP[i][0], state::Ket1().getInitialEdge());
                        }
                    }else {
                        if (index == 0){
                            partialPreSWAP[i][0] = state::Ket0().getInitialEdge();
                        }else {
                            partialPreSWAP[i][0] = mathUtils::kron(partialPreSWAP[i][0], state::Ket0().getInitialEdge());
                        }
                    }
                    if (j & 1){
                        if (index == 0){
                            partialPreSWAP[i][1] = state::Bra1().getInitialEdge();
                        }else {
                            partialPreSWAP[i][1] = mathUtils::kron(partialPreSWAP[i][1], state::Bra1().getInitialEdge());
                        }
                    } else {
                        if (index == 0){
                            partialPreSWAP[i][1] = state::Bra0().getInitialEdge();
                        }else {
                            partialPreSWAP[i][1] = mathUtils::kron(partialPreSWAP[i][1], state::Bra0().getInitialEdge());
                        }
                    }
                    i >>= 1;
                    j >>= 1;
                }
            }
            for (size_t i = 0; i < partialPreSWAP.size(); i++){
                partialSWAP[i] = mathUtils::mul(partialPreSWAP[i][0], partialPreSWAP[i][1]);
            }
            customSWAP = accumulate(partialSWAP.begin() + 1, partialSWAP.end(), partialSWAP[0], mathUtils::addWrapper);
        }
        edges.push_back(customSWAP);
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addP(int qubitIndex, double phi) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::P(phi));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::P(phi).getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronForDiagonal);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addT(int qubitIndex) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::T());
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::T().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronForDiagonal);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addTdg(int qubitIndex) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::T());
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::Tdg().getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronForDiagonal);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCP(int controlIndex, int targetIndex, double phi) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CP gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        this->gateQueue.push(gate::CP(phi));
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        if (targetIndex - controlIndex == 1) {
            edges.push_back(gate::CP(phi).getInitialEdge());
        } else {
            array<QMDDEdge, 2> partialCP;
            if (minIndex == controlIndex) {
                partialCP[0] = braketZero;
                partialCP[1] = braketOne;
            } else {
                partialCP[0] = identityEdge;
                partialCP[1] = gate::P(phi).getInitialEdge();
            }
            for (int index = minIndex + 1; index <= maxIndex; index++){
                if (index == controlIndex) {
                    partialCP[0] = mathUtils::kron(partialCP[0], braketZero);
                    partialCP[1] = mathUtils::kron(partialCP[1], braketOne);
                } else if (index == targetIndex) {
                    partialCP[0] = mathUtils::kron(partialCP[0], identityEdge);
                    partialCP[1] = mathUtils::kron(partialCP[1], gate::P(phi).getInitialEdge());
                } else {
                    partialCP[0] = mathUtils::kron(partialCP[0], identityEdge);
                    partialCP[1] = mathUtils::kron(partialCP[1], identityEdge);
                }
            }
            QMDDEdge customCP = mathUtils::add(partialCP[0], partialCP[1]);
            edges.push_back(customCP);
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCS(int controlIndex, int targetIndex) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CS gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        this->gateQueue.push(gate::CS());
    }else if(controlIndex < targetIndex) {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        if (targetIndex - controlIndex == 1) {
            edges.push_back(gate::CS().getInitialEdge());
        } else {
            array<QMDDEdge, 2> partialCS;
            if (minIndex == controlIndex) {
                partialCS[0] = braketZero;
                partialCS[1] = braketOne;
            } else {
                partialCS[0] = identityEdge;
                partialCS[1] = gate::S().getInitialEdge();
            }
            for (int index = minIndex + 1; index <= maxIndex; index++){
                if (index == controlIndex) {
                    partialCS[0] = mathUtils::kron(partialCS[0], braketZero);
                    partialCS[1] = mathUtils::kron(partialCS[1], braketOne);
                } else if (index == targetIndex) {
                    partialCS[0] = mathUtils::kron(partialCS[0], identityEdge);
                    partialCS[1] = mathUtils::kron(partialCS[1], gate::S().getInitialEdge());
                } else {
                    partialCS[0] = mathUtils::kron(partialCS[0], identityEdge);
                    partialCS[1] = mathUtils::kron(partialCS[1], identityEdge);
                }
            }
            QMDDEdge customCS = mathUtils::add(partialCS[0], partialCS[1]);
            edges.push_back(customCS);
        }
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}


void QuantumCircuit::addRx(int qubitIndex, double theta) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::Rx(theta));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::Rx(theta).getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addRy(int qubitIndex, double theta) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::Ry(theta));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::Ry(theta).getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addRz(int qubitIndex, double theta) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::Rz(theta));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::Rz(theta).getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addU(int qubitIndex, double theta, double phi, double lambda) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::U(theta, phi, lambda));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::U(theta, phi, lambda).getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addU3(int qubitIndex, double theta, double phi, double lambda) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::U3(theta, phi, lambda));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::U3(theta, phi, lambda).getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addToff(vector<int>& controlIndexes, int targetIndex) {
    if (controlIndexes.size() == 0) {
        throw invalid_argument("Control indexes must not be empty.");
    }else if (numQubits < controlIndexes.size() + 1) {
        throw invalid_argument("Number of control indexes must be at most number of qubits - 1.");
    }else if (controlIndexes.size() == 1) {
        addCX(controlIndexes[0], targetIndex);
    }else {
        sort(controlIndexes.begin(), controlIndexes.end());
        int minIndex = min(*min_element(controlIndexes.begin(), controlIndexes.end()), targetIndex);
        int maxIndex = max(*max_element(controlIndexes.begin(), controlIndexes.end()), targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        vector<QMDDEdge> partialToff(controlIndexes.size() + 1, identityEdge);
        for (int i = minIndex; i <= maxIndex; i++) {
            if (i == targetIndex) {
                if (i == minIndex) {
                    partialToff[partialToff.size() - 1] = gate::X().getInitialEdge();
                }else {
                    for (int k = 0; k < partialToff.size() - 1; k++) {
                        partialToff[k] = mathUtils::kron(partialToff[k], identityEdge);
                    }
                    partialToff[partialToff.size() - 1] = mathUtils::kron(partialToff[partialToff.size() - 1], gate::X().getInitialEdge());
                }
            } else {
                for (int j = 0; j < controlIndexes.size(); j++) {
                    for (int k = 0; k < partialToff.size(); k++) {
                        if (i == controlIndexes[j]) {
                            if (i == minIndex) {
                                if (k == j) {
                                    partialToff[k] = braketZero;
                                } else if (k > j) {
                                    partialToff[k] = braketOne;
                                }
                            } else {
                                if (k == j) {
                                    partialToff[k] = mathUtils::kron(partialToff[k], braketZero);
                                } else if (k > j) {
                                    partialToff[k] = mathUtils::kron(partialToff[k], braketOne);
                                } else if (k < j) {
                                    partialToff[k] = mathUtils::kron(partialToff[k], identityEdge);
                                }
                            }
                        }
                    }
                }
            }
        }
        QMDDEdge customToff = accumulate(partialToff.begin() + 1, partialToff.end(), partialToff[0], mathUtils::addWrapper);
        edges.push_back(customToff);
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
        return;
    }
}

void QuantumCircuit::addGate(int qubitIndex, const QMDDGate& gate) {
    if (numQubits == 1) {
        this->gateQueue.push(gate);
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate.getInitialEdge());
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kronWrapper);
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addOracle(int omega) {
    size_t numIndex;
    if (omega == 0) {
        numIndex = 1;
    } else {
        numIndex = static_cast<size_t>(log2(omega)) + 1;
    }

    bitset<64> bits(omega);
    vector<int> xIndicies
    for (int i = 0; i < numIndex; ++i) {
        if (bits[i] == 0)xIndicies.push_back(i);
    }
    if (!xIndicies.empty()) {
        this->addX(xIndicies);
    }

    vector<QMDDEdge> customI(numIndex, identityEdge);
    QMDDEdge partialCZ1 = accumulate(customI.begin() + 1, customI.end(), customI[0], mathUtils::kronWrapper);
    vector<QMDDEdge> customBrkt(numIndex, braketZero);
    QMDDEdge partialCZ2 = QMDDEdge(-2.0, accumulate(customBrkt.begin() + 1, customBrkt.end(), customBrkt[0], mathUtils::kronWrapper).uniqueTableKey);
    QMDDEdge customCZ = mathUtils::add(partialCZ1, partialCZ2);
    this->gateQueue.push(QMDDGate(customCZ));

    if (!xIndicies.empty()) {
        this->addX(xIndicies);
    }

    return;
}

void QuantumCircuit::addIAM() {
    this->addAllH();
    this->addAllX();

    vector<QMDDEdge> customI(this->numQubits, identityEdge);
    QMDDEdge partialCZ1 = accumulate(customI.begin() + 1, customI.end(), customI[0], mathUtils::kronWrapper);
    vector<QMDDEdge> customBrkt(this->numQubits, braketZero);
    QMDDEdge partialCZ2 = QMDDEdge(-2.0, accumulate(customBrkt.begin() + 1, customBrkt.end(), customBrkt[0], mathUtils::kronWrapper).uniqueTableKey);
    QMDDEdge customCZ = mathUtils::add(partialCZ1, partialCZ2);
    this->gateQueue.push(QMDDGate(customCZ));

    this->addAllX();
    this->addAllH();
    return;
}

void QuantumCircuit::simulate() {

    // OperationCache::getInstance().clearAllCaches();
    int i = 0;
    while (!this->gateQueue.empty()) {
        cout << "number of gates: " << i++ << endl;
        QMDDGate currentGate = this->gateQueue.front();
        cout << "Current gate: " << currentGate << endl;
        cout << "Current state: " << this->finalState << endl;

        cout << "============================================================\n" << endl;
        this->gateQueue.pop();
        this->finalState = QMDDState(mathUtils::mul(currentGate.getInitialEdge(), this->finalState.getInitialEdge()));
    }
    cout << "Final state: " << this->finalState << endl;
    return;
}

int QuantumCircuit::measure(int qubitIndex) {
    this->simulate();
    vector<QMDDEdge> edges0(qubitIndex, identityEdge);
    vector<QMDDEdge> edges1(qubitIndex, identityEdge);
    edges0.push_back(braketZero);
    edges1.push_back(braketOne);
    edges0.insert(edges0.end(), numQubits - qubitIndex - 1, identityEdge);
    edges1.insert(edges1.end(), numQubits - qubitIndex - 1, identityEdge);
    QMDDGate m0 = accumulate(edges0.begin() + 1, edges0.end(), edges0[0], mathUtils::kronWrapper);
    QMDDGate m1 = accumulate(edges1.begin() + 1, edges1.end(), edges1[0], mathUtils::kronWrapper);
    QMDDEdge result0 = mathUtils::mul(m0.getInitialEdge(), finalState.getInitialEdge());
    QMDDEdge result1 = mathUtils::mul(m1.getInitialEdge(), finalState.getInitialEdge());

    vector<complex<double>> v0 = result0.getAllElementsForKet();
    vector<complex<double>> v1 = result1.getAllElementsForKet();

    double p0 = mathUtils::sumOfSquares(v0);
    double p1 = mathUtils::sumOfSquares(v1);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 1.0);
    double random_value = dist(gen);

    if (random_value < p0) {
        this->finalState = QMDDState(QMDDEdge(result0.weight * (1.0 / sqrt(p0)), make_shared<QMDDNode>(*result0.getStartNode())));
        return 0;
    } else {
        this->finalState = QMDDState(QMDDEdge(result0.weight * (1.0 / sqrt(p1)), make_shared<QMDDNode>(*result1.getStartNode())));
        return 1;
    }
}