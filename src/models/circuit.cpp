#include "circuit.hpp"

static const QMDDEdge braketZero = mathUtils::mul(state::Ket0().getInitialEdge(), state::Bra0().getInitialEdge());
static const QMDDEdge braketOne = mathUtils::mul(state::Ket1().getInitialEdge(), state::Bra1().getInitialEdge());
static const QMDDEdge identityEdge = gate::I().getInitialEdge();

QuantumCircuit::QuantumCircuit(int numQubits, QMDDState initialState) : numQubits(numQubits), initialState(initialState), finalState(initialState) {
    if (numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }
}

queue<QMDDGate> QuantumCircuit::getGateQueue() const {
    return gateQueue;
}

QMDDState QuantumCircuit::getFinalState() const {
    return finalState;
}

void QuantumCircuit::addI(int qubitIndex) {
    vector<QMDDEdge> edges(numQubits);
    #pragma omp parallel for
    for (int i = 0; i < numQubits; i++) {
        edges[i] = identityEdge;
    }
    QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
    gateQueue.push(result);
    return;
}

void QuantumCircuit::addPh(int qubitIndex, double delta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Ph(delta));
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::Ph(delta).getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addX(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::X());
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::X().getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addY(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::Y());
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::Y().getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addZ(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::Z());
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::Z().getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addS(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::S());
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::S().getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addV(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::V());
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::V().getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addH(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::H());
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::H().getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
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
        cout << "minIndex: " << minIndex << endl;
        int maxIndex = max(controlIndex, targetIndex);
        cout << "maxIndex: " << maxIndex << endl;
        array<QMDDEdge, 2> partialCX;
        if (minIndex == controlIndex) {
            partialCX[0] = braketZero;
            partialCX[1] = braketOne;
        } else {
            partialCX[0] = identityEdge;
            partialCX[1] = gate::X().getInitialEdge();
        }
        vector<QMDDEdge> edges(minIndex, identityEdge);
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
        QMDDEdge customCX = mathUtils::add(partialCX[0], partialCX[1]);
        edges.push_back(customCX);
        edges.insert(edges.end(), numQubits - maxIndex - 1, identityEdge);
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
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
        array<QMDDEdge, 2> partialVarCX;
        if (minIndex == controlIndex) {
            partialVarCX[0] = braketOne;
            partialVarCX[1] = braketZero;
        } else {
            partialVarCX[0] = identityEdge;
            partialVarCX[1] = gate::X().getInitialEdge();
        }
        vector<QMDDEdge> edges(minIndex + 1, identityEdge);
        for (int index = minIndex; index <= maxIndex; index++){
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
        QMDDEdge customCX = mathUtils::add(partialVarCX[0], partialVarCX[1]);
        edges.push_back(customCX);
        edges.insert(edges.end(), numQubits - maxIndex - 1, identityEdge);
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
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
        array<QMDDEdge, 2> partialCZ;
        if (minIndex == controlIndex) {
            partialCZ[0] = braketZero;
            partialCZ[1] = braketOne;
        } else {
            partialCZ[0] = identityEdge;
            partialCZ[1] = gate::Z().getInitialEdge();
        }
        vector<QMDDEdge> edges(minIndex + 1, identityEdge);
        for (int index = minIndex; index <= maxIndex; index++){
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
        QMDDEdge customCX = mathUtils::add(partialCZ[0], partialCZ[1]);
        edges.push_back(customCX);
        edges.insert(edges.end(), numQubits - maxIndex - 1, identityEdge);
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addSWAP(int qubitIndex1, int qubitIndex2) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add SWAP gate to single qubit circuit.");
    }else if (qubitIndex1 == qubitIndex2) {
        throw invalid_argument("qubitIndexes indices must be different.");
    }else if(numQubits == 2 && ((qubitIndex1 == 0 && qubitIndex2 == 1) || (qubitIndex1 == 1 && qubitIndex2 == 0))) {
        gateQueue.push(gate::SWAP());
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
            customSWAP = accumulate(partialSWAP.begin() + 1, partialSWAP.end(), partialSWAP[0], mathUtils::add);
        }
        edges.push_back(customSWAP);
        edges.insert(edges.end(), numQubits - maxIndex - 1, identityEdge);
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

// void QuantumCircuit::addSWAP(int qubitIndex1, int qubitIndex2) {
//     if (numQubits == 1) {
//         throw invalid_argument("Cannot add SWAP gate to single qubit circuit.");
//     }else if (qubitIndex1 == qubitIndex2) {
//         throw invalid_argument("qubitIndexes indices must be different.");
//     }else if(numQubits == 2 && ((qubitIndex1 == 0 && qubitIndex2 == 1) || (qubitIndex1 == 1 && qubitIndex2 == 0))) {
//         gateQueue.push(gate::SWAP());
//     }else {
//         int minIndex = min(qubitIndex1, qubitIndex2);
//         int maxIndex = max(qubitIndex1, qubitIndex2);
//         array<array<QMDDEdge, 2>, 2> partialCX = {identityEdge, identityEdge};
//         vector<QMDDEdge> edges(minIndex, identityEdge);
//         for (int index = minIndex; index <= maxIndex; index++){
//             if (index == qubitIndex1) {
//                 if (index == minIndex) {
//                     partialCX[0][0] = braketZero;
//                     partialCX[0][1] = braketOne;
//                     partialCX[1][1] = gate::X().getInitialEdge();
//                 } else {
//                     partialCX[0][0] = mathUtils::kron(partialCX[0][0], braketZero);
//                     partialCX[0][1] = mathUtils::kron(partialCX[0][1], braketOne);
//                     partialCX[1][0] = mathUtils::kron(partialCX[1][0], identityEdge);
//                     partialCX[1][1] = mathUtils::kron(partialCX[1][1], gate::X().getInitialEdge());
//                 }
//             } else if (index == qubitIndex2) {
//                 if (index == minIndex) {
//                     partialCX[0][1] = gate::X().getInitialEdge();
//                     partialCX[1][0] = braketZero;
//                     partialCX[1][1] = braketOne;
//                 } else {
//                     partialCX[0][0] = mathUtils::kron(partialCX[0][0], identityEdge);
//                     partialCX[0][1] = mathUtils::kron(partialCX[0][1], gate::X().getInitialEdge());
//                     partialCX[1][0] = mathUtils::kron(partialCX[1][0], braketZero);
//                     partialCX[1][1] = mathUtils::kron(partialCX[1][1], braketOne);
//                 }
//             } else {
//                 for (int i = 0; i < 2; ++i) {
//                     for (int j = 0; j < 2; ++j) {
//                         partialCX[i][j] = mathUtils::kron(partialCX[i][j], identityEdge);
//                     }
//                 }
//             }
//         }
//         array<QMDDEdge,2> customCX = {mathUtils::add(partialCX[0][0], partialCX[0][1]), mathUtils::add(partialCX[1][0], partialCX[1][1])};
//         QMDDEdge customSWAP = mathUtils::mul(mathUtils::mul(customCX[0], customCX[1]), customCX[0]);
//         edges.push_back(customSWAP);
//         edges.insert(edges.end(), numQubits - maxIndex - 1, identityEdge);
//         QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
//         gateQueue.push(result);
//     }
// }

void QuantumCircuit::addP(int qubitIndex, double phi) {
    if (numQubits == 1) {
        gateQueue.push(gate::P(phi));
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::P(phi).getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addT(int qubitIndex) {
    if (numQubits == 1) {
        gateQueue.push(gate::T());
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::T().getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
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
        array<QMDDEdge, 2> partialCP;
        if (minIndex == controlIndex) {
            partialCP[0] = braketZero;
            partialCP[1] = braketOne;
        } else {
            partialCP[0] = identityEdge;
            partialCP[1] = gate::P(phi).getInitialEdge();
        }
        vector<QMDDEdge> edges(minIndex, identityEdge);
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
        QMDDEdge customCX = mathUtils::add(partialCP[0], partialCP[1]);
        edges.push_back(customCX);
        edges.insert(edges.end(), numQubits - maxIndex - 1, identityEdge);
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
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
        array<QMDDEdge, 2> partialCS;
        if (minIndex == controlIndex) {
            partialCS[0] = braketZero;
            partialCS[1] = braketOne;
        } else {
            partialCS[0] = identityEdge;
            partialCS[1] = gate::S().getInitialEdge();
        }
        vector<QMDDEdge> edges(minIndex, identityEdge);
        for (int index = minIndex; index <= maxIndex; index++){
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
        QMDDEdge customCX = mathUtils::add(partialCS[0], partialCS[1]);
        edges.push_back(customCX);
        edges.insert(edges.end(), numQubits - maxIndex - 1, identityEdge);
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}


void QuantumCircuit::addRx(int qubitIndex, double theta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Rx(theta));
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::Rx(theta).getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addRy(int qubitIndex, double theta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Ry(theta));
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::Ry(theta).getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addRz(int qubitIndex, double theta) {
    if (numQubits == 1) {
        gateQueue.push(gate::Rz(theta));
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::Rz(theta).getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addU(int qubitIndex, double theta, double phi, double lambda) {
    if (numQubits == 1) {
        gateQueue.push(gate::U(theta, phi, lambda));
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate::U(theta, phi, lambda).getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
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
        QMDDEdge customToff = accumulate(partialToff.begin() + 1, partialToff.end(), partialToff[0], mathUtils::add);
        edges.push_back(customToff);
        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
        gateQueue.push(result);
        return;
    }
}

void QuantumCircuit::addToff2(array<int, 2>& controlIndexes, int targetIndex) {
    if (numQubits < 3) {
        throw invalid_argument("Cannot add Toffoli gate to single qubit circuit.");
    }else if (controlIndexes[0] == controlIndexes[1] || controlIndexes[0] == targetIndex || controlIndexes[1] == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else {
        sort(controlIndexes.begin(), controlIndexes.end());
        if (controlIndexes[0] == 0 && controlIndexes[1] == 1 && targetIndex == 2 && numQubits == 3) {
            gateQueue.push(gate::Toff());
            return;
        }else {
            int minIndex = min(*min_element(controlIndexes.begin(), controlIndexes.end()), targetIndex);
            int maxIndex = max(*max_element(controlIndexes.begin(), controlIndexes.end()), targetIndex);
            vector<QMDDEdge> edges(minIndex, identityEdge);
            QMDDEdge customH = identityEdge;
            array<QMDDEdge, 2> partialCX23 = {identityEdge, identityEdge};
            QMDDEdge customTdagger3 = identityEdge;
            array<QMDDEdge, 2> partialCX13 = {identityEdge, identityEdge};
            QMDDEdge customT3 = identityEdge;
            QMDDEdge customT2 = identityEdge;
            array<QMDDEdge, 2> partialCX12 = {identityEdge, identityEdge};
            QMDDEdge customT1Tdagger2 = identityEdge;
            for (int i = minIndex; i <= maxIndex; i++) {
                if (i == controlIndexes[0]) {
                    if (i == minIndex) {
                        partialCX13[0] = braketZero;
                        partialCX13[1] = braketOne;
                        partialCX12[0] = braketZero;
                        partialCX12[1] = braketOne;
                        customT1Tdagger2 = gate::T().getInitialEdge();
                    } else {
                        customH = mathUtils::kron(customH, identityEdge);
                        partialCX23[0] = mathUtils::kron(partialCX23[0], identityEdge);
                        partialCX23[1] = mathUtils::kron(partialCX23[1], identityEdge);
                        customTdagger3 = mathUtils::kron(customTdagger3, identityEdge);
                        partialCX13[0] = mathUtils::kron(partialCX13[0], braketZero);
                        partialCX13[1] = mathUtils::kron(partialCX13[1], braketOne);
                        customT3 = mathUtils::kron(customT3, identityEdge);
                        customT2 = mathUtils::kron(customT2, identityEdge);
                        partialCX12[0] = mathUtils::kron(partialCX12[0], braketZero);
                        partialCX12[1] = mathUtils::kron(partialCX12[1], braketOne);
                        customT1Tdagger2 = mathUtils::kron(customT1Tdagger2, gate::T().getInitialEdge());
                    }
                } else if (i == controlIndexes[1]) {
                    if (i == minIndex) {
                        partialCX23[0] = braketZero;
                        partialCX23[1] = braketOne;
                        customT2 = gate::T().getInitialEdge();
                        partialCX12[1] = gate::X().getInitialEdge();
                        customT1Tdagger2 = gate::Tdagger().getInitialEdge();
                    } else {
                        customH = mathUtils::kron(customH, identityEdge);
                        partialCX23[0] = mathUtils::kron(partialCX23[0], braketZero);
                        partialCX23[1] = mathUtils::kron(partialCX23[1], braketOne);
                        customTdagger3 = mathUtils::kron(customTdagger3, identityEdge);
                        partialCX13[0] = mathUtils::kron(partialCX13[0], identityEdge);
                        partialCX13[1] = mathUtils::kron(partialCX13[1], identityEdge);
                        customT3 = mathUtils::kron(customT3, identityEdge);
                        customT2 = mathUtils::kron(customT2, gate::T().getInitialEdge());
                        partialCX12[0] = mathUtils::kron(partialCX12[0], identityEdge);
                        partialCX12[1] = mathUtils::kron(partialCX12[1], gate::X().getInitialEdge());
                        customT1Tdagger2 = mathUtils::kron(customT1Tdagger2, gate::Tdagger().getInitialEdge());
                    }
                } else if (i == targetIndex) {
                    if (i == minIndex) {
                        customH = gate::H().getInitialEdge();
                        partialCX23[1] = gate::X().getInitialEdge();
                        customTdagger3 = gate::Tdagger().getInitialEdge();
                        partialCX13[1] = gate::X().getInitialEdge();
                        customT3 = gate::T().getInitialEdge();
                    } else {
                        customH = mathUtils::kron(customH, gate::H().getInitialEdge());
                        partialCX23[0] = mathUtils::kron(partialCX23[0], identityEdge);
                        partialCX23[1] = mathUtils::kron(partialCX23[1], gate::X().getInitialEdge());
                        customTdagger3 = mathUtils::kron(customTdagger3, gate::Tdagger().getInitialEdge());
                        partialCX13[0] = mathUtils::kron(partialCX13[0], identityEdge);
                        partialCX13[1] = mathUtils::kron(partialCX13[1], gate::X().getInitialEdge());
                        customT3 = mathUtils::kron(customT3, gate::T().getInitialEdge());
                        customT2 = mathUtils::kron(customT2, identityEdge);
                        partialCX12[0] = mathUtils::kron(partialCX12[0], identityEdge);
                        partialCX12[1] = mathUtils::kron(partialCX12[1], identityEdge);
                        customT1Tdagger2 = mathUtils::kron(customT1Tdagger2, identityEdge);
                    }
                } else {
                    customH = mathUtils::kron(customH, identityEdge);
                    partialCX23[0] = mathUtils::kron(partialCX23[0], identityEdge);
                    partialCX23[1] = mathUtils::kron(partialCX23[1], identityEdge);
                    customTdagger3 = mathUtils::kron(customTdagger3, identityEdge);
                    partialCX13[0] = mathUtils::kron(partialCX13[0], identityEdge);
                    partialCX13[1] = mathUtils::kron(partialCX13[1], identityEdge);
                    customT3 = mathUtils::kron(customT3, identityEdge);
                    customT2 = mathUtils::kron(customT2, identityEdge);
                    partialCX12[0] = mathUtils::kron(partialCX12[0], identityEdge);
                    partialCX12[1] = mathUtils::kron(partialCX12[1], identityEdge);
                    customT1Tdagger2 = mathUtils::kron(customT1Tdagger2, identityEdge);
                }
            }
            QMDDEdge customCX23 = mathUtils::add(partialCX23[0], partialCX23[1]);
            QMDDEdge customCX13 = mathUtils::add(partialCX13[0], partialCX13[1]);
            QMDDEdge customCX12 = mathUtils::add(partialCX12[0], partialCX12[1]);
            QMDDEdge partialToff2 = mathUtils::mul(mathUtils::mul(mathUtils::mul(customCX23, customTdagger3), customCX13), customT3);
            QMDDEdge customToff2 = mathUtils::mul(mathUtils::mul(mathUtils::mul(mathUtils::mul(mathUtils::mul(mathUtils::mul(mathUtils::mul(customH, partialToff2), partialToff2), customT2), customCX12), customH), customT1Tdagger2), customCX12);
            edges.push_back(customToff2);
            edges.insert(edges.end(), numQubits - maxIndex - 1, identityEdge);
            QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
            gateQueue.push(result);
            return;
        }
    }
}

void QuantumCircuit::addGate(int qubitIndex, const QMDDGate& gate) {
    if (numQubits == 1) {
        gateQueue.push(gate);
    } else {
        vector<QMDDEdge> edges(numQubits);

        #pragma omp parallel for
        for (int i = 0; i < numQubits; i++) {
            if (i == qubitIndex) {
                edges[i] = gate.getInitialEdge();
            } else {
                edges[i] = identityEdge;
            }
        }

        QMDDGate result = accumulate(edges.begin() + 1, edges.end(), edges[0], mathUtils::kron);
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
        gateQueue.pop();
        currentState = mathUtils::mul(currentGate.getInitialEdge(), currentState.getInitialEdge());
    }
    finalState = currentState;
    cout << "Final state: " << finalState << endl;
    return;
}

QMDDState QuantumCircuit::read(int qubitIndex) {
    QuantumCircuit::execute();
    vector<QMDDEdge> edges0(qubitIndex, identityEdge);
    vector<QMDDEdge> edges1(qubitIndex, identityEdge);
    edges0.push_back(braketZero);
    edges1.push_back(braketOne);
    edges0.insert(edges0.end(), numQubits - qubitIndex - 1, identityEdge);
    edges1.insert(edges1.end(), numQubits - qubitIndex - 1, identityEdge);
    QMDDGate m0 = accumulate(edges0.begin() + 1, edges0.end(), edges0[0], mathUtils::kron);
    QMDDGate m1 = accumulate(edges1.begin() + 1, edges1.end(), edges1[0], mathUtils::kron);
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
        finalState = QMDDState(QMDDEdge(result0.weight * (1.0 / sqrt(p0)), make_shared<QMDDNode>(*result0.getStartNode())));
        return state::Ket0();
    } else {
        finalState = QMDDState(QMDDEdge(result0.weight * (1.0 / sqrt(p1)), make_shared<QMDDNode>(*result1.getStartNode())));
        return state::Ket1();
    }
}