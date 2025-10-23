#include "circuit.hpp"


QuantumCircuit::QuantumCircuit(int numQubits, QMDDState initialState) : numQubits(numQubits), finalState(initialState) {
    call_once(initExtendedEdgeFlag, initExtendedEdge);
    if (this->numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }
    this->quantumRegister.resize(1);
    this->setRegister(0, this->numQubits);
}

QuantumCircuit::QuantumCircuit(int numQubits) : numQubits(numQubits), finalState(state::Ket0()) {

    call_once(initExtendedEdgeFlag, initExtendedEdge);
    if (this->numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }

    for (int i = 1; i < this->numQubits; i++) {
        this->finalState = mathUtils::kron(state::Ket0().getInitialEdge(), this->finalState.getInitialEdge());
    }
    this->quantumRegister.resize(1);
    this->setRegister(0, this->numQubits);
}

queue<QMDDGate> QuantumCircuit::getGateQueue() const {
    return this->gateQueue;
}

QMDDState QuantumCircuit::getFinalState() const {
    return this->finalState;
}

void QuantumCircuit::setRegister(int registerIdx, int size) {
    if (registerIdx < 0) {
        throw out_of_range("Invalid register index.");
    }

    if (registerIdx >= static_cast<int>(this->quantumRegister.size())) {
        this->quantumRegister.resize(registerIdx + 1);
    }

    this->quantumRegister[registerIdx].resize(size);
    iota(this->quantumRegister[registerIdx].begin(), this->quantumRegister[registerIdx].end(), registerIdx == 0 ? 0 : this->quantumRegister[registerIdx - 1].back() + 1);
}

void QuantumCircuit::addI(int qubitIndex) {
    return;
}

void QuantumCircuit::addPh(int qubitIndex, double delta) {
    if (numQubits == 1 || qubitIndex == 0) {
        this->gateQueue.push(gate::Ph(delta));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::Ph(delta).getInitialEdge());

        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });

        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addX(int qubitIndex) {
    if (numQubits == 1 || qubitIndex == 0) {
        this->gateQueue.push(gate::X());
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::X().getInitialEdge());

        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });

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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addAllX() {
    vector<QMDDEdge> edges(this->numQubits, gate::X().getInitialEdge());
    QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    });
    this->gateQueue.push(result);
    return;
}

void QuantumCircuit::addY(int qubitIndex) {
    if (numQubits == 1 || qubitIndex == 0) {
        this->gateQueue.push(gate::Y());
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::Y().getInitialEdge());

        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });

        this->gateQueue.push(result);
    }
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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addZ(int qubitIndex) {
    if (numQubits == 1 || qubitIndex == 0) {
        this->gateQueue.push(gate::Z());
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::Z().getInitialEdge());

        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });

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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addS(int qubitIndex) {
    if (numQubits == 1 || qubitIndex == 0) {
        this->gateQueue.push(gate::S());
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::S().getInitialEdge());

        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });

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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addSdg(int qubitIndex) {
    if (numQubits == 1 || qubitIndex == 0) {
        this->gateQueue.push(gate::Sdg());
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::Sdg().getInitialEdge());

        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });

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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addV(int qubitIndex) {
    if (numQubits == 1 || qubitIndex == 0) {
        this->gateQueue.push(gate::V());
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::V().getInitialEdge());

        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });

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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addH(int qubitIndex) {
    if (numQubits == 1 || qubitIndex == 0) {
        this->gateQueue.push(gate::H());
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::H().getInitialEdge());

        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });

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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addAllH() {
    for (size_t i = 0; i < this->numQubits; i++) {
        this->addH(i);
    }
    return;
}

void QuantumCircuit::addCX(int controlIndex, int targetIndex) {
    if (this->numQubits == 1) {
        throw invalid_argument("Cannot add CX gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 2 && controlIndex == 0 && targetIndex == 1) {
        this->gateQueue.push(gate::CX1());
    }else {
        if (controlIndex > targetIndex) {
            this->addH(targetIndex);
            this->addCZ(controlIndex, targetIndex);
            this->addH(targetIndex);
        } else {
            QMDDEdge customCX;
            vector<QMDDEdge> edges(controlIndex, identityEdge);
            if (targetIndex - controlIndex == 1) {
                customCX = gate::CX1().getInitialEdge();
            } else {
                array<QMDDEdge, 2> partialCX = {identityEdge, gate::X().getInitialEdge()};
                for ([[maybe_unused]] int _ = targetIndex - 1; _ > controlIndex; _--) {
                        partialCX[0] = mathUtils::kron(identityEdge, partialCX[0]);
                        partialCX[1] = mathUtils::kron(identityEdge, partialCX[1]);
                }
                customCX = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                    {partialCX[0], edgeZero},
                    {edgeZero, partialCX[1]}
                }));
            }
            edges.push_back(customCX);
            QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
                return mathUtils::kron(current, accumulated);
            });
            this->gateQueue.push(result);
        }
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
            if (maxIndex == controlIndex) {
                partialVarCX[0] = braketZero;
                partialVarCX[1] = braketOne;
            } else {
                partialVarCX[0] = gate::X().getInitialEdge();
                partialVarCX[1] = identityEdge;
            }
            for (int index = maxIndex - 1; index >= minIndex; index--) {
                if (index == controlIndex) {
                    partialVarCX[0] = mathUtils::kron(braketZero, partialVarCX[0]);
                    partialVarCX[1] = mathUtils::kron(braketOne, partialVarCX[1]);
                } else if (index == targetIndex) {
                    partialVarCX[0] = mathUtils::kron(gate::X().getInitialEdge(), partialVarCX[0]);
                    partialVarCX[1] = mathUtils::kron(identityEdge, partialVarCX[1]);
                } else {
                    partialVarCX[0] = mathUtils::kron(identityEdge, partialVarCX[0]);
                    partialVarCX[1] = mathUtils::kron(identityEdge, partialVarCX[1]);
                }
            }
            QMDDEdge customVarCX = mathUtils::add(partialVarCX[0], partialVarCX[1]);
            edges.push_back(customVarCX);
        }
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCY(int controlIndex, int targetIndex) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CY gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        array<QMDDEdge, 2> partialCY;
        if (maxIndex == controlIndex) {
            partialCY[0] = braketZero;
            partialCY[1] = braketOne;
        } else {
            partialCY[0] = identityEdge;
            partialCY[1] = gate::Y().getInitialEdge();
        }
        for (int index = maxIndex - 1; index >= minIndex; index--) {
            if (index == controlIndex) {
                partialCY[0] = mathUtils::kron(braketZero, partialCY[0]);
                partialCY[1] = mathUtils::kron(braketOne, partialCY[1]);
            } else if (index == targetIndex) {
                partialCY[0] = mathUtils::kron(identityEdge, partialCY[0]);
                partialCY[1] = mathUtils::kron(gate::Y().getInitialEdge(), partialCY[1]);
            } else {
                partialCY[0] = mathUtils::kron(identityEdge, partialCY[0]);
                partialCY[1] = mathUtils::kron(identityEdge, partialCY[1]);
            }
        }
        QMDDEdge customCY = mathUtils::add(partialCY[0], partialCY[1]);
        edges.push_back(customCY);
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
        if (maxIndex - minIndex == 1) {
            edges.push_back(gate::CZ().getInitialEdge());
        } else {
            array<QMDDEdge, 2> partialCZ;
            if (maxIndex == controlIndex) {
                partialCZ[0] = braketZero;
                partialCZ[1] = braketOne;
            } else {
                partialCZ[0] = identityEdge;
                partialCZ[1] = gate::Z().getInitialEdge();
            }
            for (int index = maxIndex - 1; index >= minIndex; index--) {
                if (index == controlIndex) {
                    partialCZ[0] = mathUtils::kron(braketZero, partialCZ[0]);
                    partialCZ[1] = mathUtils::kron(braketOne, partialCZ[1]);
                } else if (index == targetIndex) {
                    partialCZ[0] = mathUtils::kron(identityEdge, partialCZ[0]);
                    partialCZ[1] = mathUtils::kron(gate::Z().getInitialEdge(), partialCZ[1]);
                } else {
                    partialCZ[0] = mathUtils::kron(identityEdge, partialCZ[0]);
                    partialCZ[1] = mathUtils::kron(identityEdge, partialCZ[1]);
                }
            }
            QMDDEdge customCZ = mathUtils::add(partialCZ[0], partialCZ[1]);
            edges.push_back(customCZ);
        }
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
                int highestBit = (i >> (numIndex - 1)) & 1;
                int lowestBit = i & 1;
                int basedIndex = i;
                int swappedIndex = (i & ~(1ULL << (numIndex - 1))) | (lowestBit << (numIndex - 1));
                swappedIndex = (swappedIndex & ~1ULL) | highestBit;
                for ([[maybe_unused]] int _ = numIndex - 1; _ >= 0; _--) {
                    bool msbBased = (basedIndex >> (numIndex - 1)) & 1;
                    bool msbSwapped = (swappedIndex >> (numIndex - 1)) & 1;
                    if (msbBased){
                        if (_ == numIndex - 1){
                            partialPreSWAP[i][0] = state::Ket1().getInitialEdge();
                        }else {
                            partialPreSWAP[i][0] = mathUtils::kron(state::Ket1().getInitialEdge(), partialPreSWAP[i][0]);
                        }
                    }else {
                        if (_ == numIndex - 1){
                            partialPreSWAP[i][0] = state::Ket0().getInitialEdge();
                        }else {
                            partialPreSWAP[i][0] = mathUtils::kron(state::Ket0().getInitialEdge(), partialPreSWAP[i][0]);
                        }
                    }
                    if (msbSwapped){
                        if (_ == numIndex - 1){
                            partialPreSWAP[i][1] = state::Bra1().getInitialEdge();
                        }else {
                            partialPreSWAP[i][1] = mathUtils::kron(state::Bra1().getInitialEdge(), partialPreSWAP[i][1]);
                        }
                    } else {
                        if (_ == numIndex - 1){
                            partialPreSWAP[i][1] = state::Bra0().getInitialEdge();
                        }else {
                            partialPreSWAP[i][1] = mathUtils::kron(state::Bra0().getInitialEdge(), partialPreSWAP[i][1]);
                        }
                    }
                    basedIndex <<= 1;
                    swappedIndex <<= 1;
                }
                partialSWAP[i] = mathUtils::dyad(partialPreSWAP[i][0], partialPreSWAP[i][1]);
            }
            customSWAP = accumulate(partialSWAP.begin() + 1, partialSWAP.end(), partialSWAP[0], [](const QMDDEdge& accumulated, const QMDDEdge& current) {
                return mathUtils::add(accumulated, current);
            });
        }
        edges.push_back(customSWAP);
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
            if (maxIndex == controlIndex) {
                partialCP[0] = braketZero;
                partialCP[1] = braketOne;
            } else {
                partialCP[0] = identityEdge;
                partialCP[1] = gate::P(phi).getInitialEdge();
            }
            for (int index = maxIndex - 1; index >= minIndex; index--) {
                if (index == controlIndex) {
                    partialCP[0] = mathUtils::kron(braketZero, partialCP[0]);
                    partialCP[1] = mathUtils::kron(braketOne, partialCP[1]);
                } else if (index == targetIndex) {
                    partialCP[0] = mathUtils::kron(identityEdge, partialCP[0]);
                    partialCP[1] = mathUtils::kron(gate::P(phi).getInitialEdge(), partialCP[1]);
                } else {
                    partialCP[0] = mathUtils::kron(identityEdge, partialCP[0]);
                    partialCP[1] = mathUtils::kron(identityEdge, partialCP[1]);
                }
            }
            QMDDEdge customCP = mathUtils::add(partialCP[0], partialCP[1]);
            edges.push_back(customCP);
        }
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
            if (maxIndex == controlIndex) {
                partialCS[0] = braketZero;
                partialCS[1] = braketOne;
            } else {
                partialCS[0] = identityEdge;
                partialCS[1] = gate::S().getInitialEdge();
            }
            for (int index = maxIndex - 1; index >= minIndex; index--) {
                if (index == controlIndex) {
                    partialCS[0] = mathUtils::kron(braketZero, partialCS[0]);
                    partialCS[1] = mathUtils::kron(braketOne, partialCS[1]);
                } else if (index == targetIndex) {
                    partialCS[0] = mathUtils::kron(identityEdge, partialCS[0]);
                    partialCS[1] = mathUtils::kron(gate::S().getInitialEdge(), partialCS[1]);
                } else {
                    partialCS[0] = mathUtils::kron(identityEdge, partialCS[0]);
                    partialCS[1] = mathUtils::kron(identityEdge, partialCS[1]);
                }
            }
            QMDDEdge customCS = mathUtils::add(partialCS[0], partialCS[1]);
            edges.push_back(customCS);
        }
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCH(int controlIndex, int targetIndex) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CH gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        array<QMDDEdge, 2> partialCH;
        if (maxIndex == controlIndex) {
            partialCH[0] = braketZero;
            partialCH[1] = braketOne;
        } else {
            partialCH[0] = identityEdge;
            partialCH[1] = gate::H().getInitialEdge();
        }
        for (int index = maxIndex - 1; index >= minIndex; index--) {
            if (index == controlIndex) {
                partialCH[0] = mathUtils::kron(braketZero, partialCH[0]);
                partialCH[1] = mathUtils::kron(braketOne, partialCH[1]);
            } else if (index == targetIndex) {
                partialCH[0] = mathUtils::kron(identityEdge, partialCH[0]);
                partialCH[1] = mathUtils::kron(gate::H().getInitialEdge(), partialCH[1]);
            } else {
                partialCH[0] = mathUtils::kron(identityEdge, partialCH[0]);
                partialCH[1] = mathUtils::kron(identityEdge, partialCH[1]);
            }
        }
        QMDDEdge customCH = mathUtils::add(partialCH[0], partialCH[1]);
        edges.push_back(customCH);
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addFREDKIN(int controlIndex, int targetIndex1, int targetIndex2) {
    if (numQubits < 3) {
        throw invalid_argument("Cannot add Fredkin gate to less than 3 qubit circuit.");
    }else if (controlIndex == targetIndex1 || controlIndex == targetIndex2 || targetIndex1 == targetIndex2) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 3 && ((controlIndex == 0 && targetIndex1 == 1 && targetIndex2 == 2) || (controlIndex == 0 && targetIndex1 == 2 && targetIndex2 == 1))) {
        this->gateQueue.push(gate::FREDKIN());
    }else {
        int minTargetIndex = min(targetIndex1, targetIndex2);
        int maxTargetIndex = max(targetIndex1, targetIndex2);
        int minIndex = min(controlIndex, minTargetIndex);
        int maxIndex = max(controlIndex, maxTargetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        array<QMDDEdge, 2> partialFredkin;
        if (maxIndex == controlIndex) {
            partialFredkin[0] = braketZero;
            partialFredkin[1] = braketOne;
        } else {
            partialFredkin[0] = identityEdge;
        }
        for (int index = maxIndex - 1; index >= minIndex; index--) {
            if (index == controlIndex) {
                partialFredkin[0] = mathUtils::kron(braketZero, partialFredkin[0]);
                partialFredkin[1] = mathUtils::kron(braketOne, partialFredkin[1]);
            } else if (index == targetIndex1) {
                partialFredkin[0] = mathUtils::kron(identityEdge, partialFredkin[0]);
            }
        }
        QMDDEdge customFredkin;
        customFredkin = mathUtils::add(partialFredkin[0], partialFredkin[1]);
    }
}

void QuantumCircuit::addU(int qubitIndex, double theta, double phi, double lambda) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::U(theta, phi, lambda));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::U(theta, phi, lambda).getInitialEdge());
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addU1(int qubitIndex, double theta) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::U1(theta));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::U1(theta).getInitialEdge());
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addU2(int qubitIndex, double phi, double lambda) {
    if (numQubits == 1) {
        this->gateQueue.push(gate::U2(phi, lambda));
    } else {
        vector<QMDDEdge> edges(qubitIndex, identityEdge);
        edges.push_back(gate::U2(phi, lambda).getInitialEdge());
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCRx(int controlIndex, int targetIndex, double theta) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CRX gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        array<QMDDEdge, 2> partialCRx;
        if (maxIndex == controlIndex) {
            partialCRx[0] = braketZero;
            partialCRx[1] = braketOne;
        } else {
            partialCRx[0] = identityEdge;
            partialCRx[1] = gate::Rx(theta).getInitialEdge();
        }
        for (int index = maxIndex - 1; index >= minIndex; index--) {
            if (index == controlIndex) {
                partialCRx[0] = mathUtils::kron(braketZero, partialCRx[0]);
                partialCRx[1] = mathUtils::kron(braketOne, partialCRx[1]);
            } else if (index == targetIndex) {
                partialCRx[0] = mathUtils::kron(identityEdge, partialCRx[0]);
                partialCRx[1] = mathUtils::kron(gate::Rx(theta).getInitialEdge(), partialCRx[1]);
            } else {
                partialCRx[0] = mathUtils::kron(identityEdge, partialCRx[0]);
                partialCRx[1] = mathUtils::kron(identityEdge, partialCRx[1]);
            }
        }
        QMDDEdge customCRx = mathUtils::add(partialCRx[0], partialCRx[1]);
        edges.push_back(customCRx);
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCRy(int controlIndex, int targetIndex, double theta) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CRY gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        array<QMDDEdge, 2> partialCRy;
        if (maxIndex == controlIndex) {
            partialCRy[0] = braketZero;
            partialCRy[1] = braketOne;
        } else {
            partialCRy[0] = identityEdge;
            partialCRy[1] = gate::Ry(theta).getInitialEdge();
        }
        for (int index = maxIndex - 1; index >= minIndex; index--) {
            if (index == controlIndex) {
                partialCRy[0] = mathUtils::kron(braketZero, partialCRy[0]);
                partialCRy[1] = mathUtils::kron(braketOne, partialCRy[1]);
            } else if (index == targetIndex) {
                partialCRy[0] = mathUtils::kron(identityEdge, partialCRy[0]);
                partialCRy[1] = mathUtils::kron(gate::Ry(theta).getInitialEdge(), partialCRy[1]);
            } else {
                partialCRy[0] = mathUtils::kron(identityEdge, partialCRy[0]);
                partialCRy[1] = mathUtils::kron(identityEdge, partialCRy[1]);
            }
        }
        QMDDEdge customCRy = mathUtils::add(partialCRy[0], partialCRy[1]);
        edges.push_back(customCRy);
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCRz(int controlIndex, int targetIndex, double theta) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CRZ gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        array<QMDDEdge, 2> partialCRz;
        if (maxIndex == controlIndex) {
            partialCRz[0] = braketZero;
            partialCRz[1] = braketOne;
        } else {
            partialCRz[0] = identityEdge;
            partialCRz[1] = gate::Rz(theta).getInitialEdge();
        }
        for (int index = maxIndex - 1; index >= minIndex; index--) {
            if (index == controlIndex) {
                partialCRz[0] = mathUtils::kron(braketZero, partialCRz[0]);
                partialCRz[1] = mathUtils::kron(braketOne, partialCRz[1]);
            } else if (index == targetIndex) {
                partialCRz[0] = mathUtils::kron(identityEdge, partialCRz[0]);
                partialCRz[1] = mathUtils::kron(gate::Rz(theta).getInitialEdge(), partialCRz[1]);
            } else {
                partialCRz[0] = mathUtils::kron(identityEdge, partialCRz[0]);
                partialCRz[1] = mathUtils::kron(identityEdge, partialCRz[1]);
            }
        }
        QMDDEdge customCRz = mathUtils::add(partialCRz[0], partialCRz[1]);
        edges.push_back(customCRz);
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addCU(int controlIndex, int targetIndex, double theta, double phi, double lambda) {
    if (numQubits == 1) {
        throw invalid_argument("Cannot add CU gate to single qubit circuit.");
    }else if (controlIndex == targetIndex) {
        throw invalid_argument("Control and target indices must be different.");
    }else {
        int minIndex = min(controlIndex, targetIndex);
        int maxIndex = max(controlIndex, targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        array<QMDDEdge, 2> partialCU;
        if (maxIndex == controlIndex) {
            partialCU[0] = braketZero;
            partialCU[1] = braketOne;
        } else {
            partialCU[0] = identityEdge;
            partialCU[1] = gate::U(theta, phi, lambda).getInitialEdge();
        }
        for (int index = maxIndex - 1; index >= minIndex; index--) {
            if (index == controlIndex) {
                partialCU[0] = mathUtils::kron(braketZero, partialCU[0]);
                partialCU[1] = mathUtils::kron(braketOne, partialCU[1]);
            } else if (index == targetIndex) {
                partialCU[0] = mathUtils::kron(identityEdge, partialCU[0]);
                partialCU[1] = mathUtils::kron(gate::U(theta, phi, lambda).getInitialEdge(), partialCU[1]);
            } else {
                partialCU[0] = mathUtils::kron(identityEdge, partialCU[0]);
                partialCU[1] = mathUtils::kron(identityEdge, partialCU[1]);
            }
        }
        QMDDEdge customCU = mathUtils::add(partialCU[0], partialCU[1]);
        edges.push_back(customCU);
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addToff(const array<int, 2>& controlIndexes, int targetIndex) {
    if (controlIndexes.size() == 0) {
        throw invalid_argument("Control indexes must not be empty.");
    }else if (numQubits < controlIndexes.size() + 1) {
        throw invalid_argument("Number of control indexes must be at most number of qubits - 1.");
    }else if (controlIndexes.size() == 1) {
        addCX(controlIndexes[0], targetIndex);
    }else {
        array<int, 2> sortedControlIndexes = sorted(controlIndexes);
        int minIndex = min(*min_element(sortedControlIndexes.begin(), sortedControlIndexes.end()), targetIndex);
        int maxIndex = max(*max_element(sortedControlIndexes.begin(), sortedControlIndexes.end()), targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        vector<QMDDEdge> partialToff(sortedControlIndexes.size() + 1, identityEdge);
        for (int i = maxIndex; i >= minIndex; i--) {
            if (i == targetIndex) {
                if (i == maxIndex) {
                    partialToff[partialToff.size() - 1] = gate::X().getInitialEdge();
                }else {
                    for (int j = 0; j < partialToff.size() - 1; j++) {
                        partialToff[j] = mathUtils::kron(identityEdge, partialToff[j]);
                    }
                    partialToff[partialToff.size() - 1] = mathUtils::kron(gate::X().getInitialEdge(), partialToff[partialToff.size() - 1]);
                }
            } else {
                auto idx = ranges::find(sortedControlIndexes, i);
                if (idx != sortedControlIndexes.end()) {
                    int j = static_cast<int>(distance(sortedControlIndexes.begin(), idx));
                            for (int k = 0; k < partialToff.size(); k++) {
                                if (i == maxIndex) {
                                    if (k == j) {
                                        partialToff[k] = braketZero;
                                    } else if (k > j) {
                                        partialToff[k] = braketOne;
                                    } else if (k < j) {
                                        partialToff[k] = identityEdge;
                                    }
                                } else {
                                    if (k == j) {
                                        partialToff[k] = mathUtils::kron(braketZero, partialToff[k]);
                                    } else if (k > j) {
                                        partialToff[k] = mathUtils::kron(braketOne, partialToff[k]);
                                    } else if (k < j) {
                                        partialToff[k] = mathUtils::kron(identityEdge, partialToff[k]);
                                    }
                                }
                            }
                } else {
                    if (i != maxIndex) {
                        for (int j = 0; j < partialToff.size(); j++) {
                            partialToff[j] = mathUtils::kron(identityEdge, partialToff[j]);
                        }
                    }
                }
            }
        }
        QMDDEdge customToff = accumulate(partialToff.begin() + 1, partialToff.end(), partialToff[0], [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::add(accumulated, current);
        });
        edges.push_back(customToff);
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);

        return;
    }
}

void QuantumCircuit::addMCT(const vector<int>& controlIndexes, int targetIndex) {
    if (controlIndexes.size() == 0) {
        throw invalid_argument("Control indexes must not be empty.");
    }else if (numQubits < controlIndexes.size() + 1) {
        throw invalid_argument("Number of control indexes must be at most number of qubits - 1.");
    }else if (controlIndexes.size() == 1) {
        addCX(controlIndexes[0], targetIndex);
    }else {
        vector<int> sortedControlIndexes = sorted(controlIndexes);
        int minIndex = min(*min_element(sortedControlIndexes.begin(), sortedControlIndexes.end()), targetIndex);
        int maxIndex = max(*max_element(sortedControlIndexes.begin(), sortedControlIndexes.end()), targetIndex);
        vector<QMDDEdge> edges(minIndex, identityEdge);
        vector<QMDDEdge> partialMCT(sortedControlIndexes.size() + 1, identityEdge);
        for (int i = maxIndex; i >= minIndex; i--) {
            if (i == targetIndex) {
                if (i == maxIndex) {
                    partialMCT[partialMCT.size() - 1] = gate::X().getInitialEdge();
                }else {
                    for (int j = 0; j < partialMCT.size() - 1; j++) {
                        partialMCT[j] = mathUtils::kron(identityEdge, partialMCT[j]);
                    }
                    partialMCT[partialMCT.size() - 1] = mathUtils::kron(gate::X().getInitialEdge(), partialMCT[partialMCT.size() - 1]);
                }
            } else {
                auto idx = ranges::find(sortedControlIndexes, i);
                if (idx != sortedControlIndexes.end()) {
                    int j = static_cast<int>(distance(sortedControlIndexes.begin(), idx));
                            for (int k = 0; k < partialMCT.size(); k++) {
                                if (i == maxIndex) {
                                    if (k == j) {
                                        partialMCT[k] = braketZero;
                                    } else if (k > j) {
                                        partialMCT[k] = braketOne;
                                    } else if (k < j) {
                                        partialMCT[k] = identityEdge;
                                    }
                                } else {
                                    if (k == j) {
                                        partialMCT[k] = mathUtils::kron(braketZero, partialMCT[k]);
                                    } else if (k > j) {
                                        partialMCT[k] = mathUtils::kron(braketOne, partialMCT[k]);
                                    } else if (k < j) {
                                        partialMCT[k] = mathUtils::kron(identityEdge, partialMCT[k]);
                                    }
                                }
                            }
                } else {
                    if (i != maxIndex) {
                        for (int j = 0; j < partialMCT.size(); j++) {
                            partialMCT[j] = mathUtils::kron(identityEdge, partialMCT[j]);
                        }
                    }
                }
            }
        }
        QMDDEdge customMCT = accumulate(partialMCT.begin() + 1, partialMCT.end(), partialMCT[0], [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::add(accumulated, current);
        });
        edges.push_back(customMCT);
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
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
        QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::kron(current, accumulated);
        });
        this->gateQueue.push(result);
    }
    return;
}

void QuantumCircuit::addBARRIER() {
    return;
}

void QuantumCircuit::addQFT(int numQubits) {
    for (int i = numQubits - 1; i >= 0; i--) {
        this->addH(i);
        for (int j = i - 1; j >= 0; j--) {
            this->addCP(i, j, M_PI / pow(2, j - i));
        }
    }

    for (int i = 0; i < numQubits / 2; i++) {
        this->addSWAP(i, numQubits - i - 1);
    }

    return;
}

void QuantumCircuit::addQFT() {
    this->addQFT(this->numQubits);
    return;
}

void QuantumCircuit::addOracle(int omega) {
    size_t numIndex = omega == 0 ? 1 : static_cast<size_t>(ceil(log2(omega + 1)));

    bitset<64> bits(omega);
    vector<QMDDEdge> customBrkt;
    for (int i = 0; i < numIndex; ++i) {
        customBrkt.push_back(bits[i] ? braketOne : braketZero);
    }

    vector<QMDDEdge> customI(numIndex, identityEdge);
    QMDDEdge partialCZ1 = accumulate(customI.rbegin() + 1, customI.rend(), customI.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    });
    QMDDEdge partialCZ2 = QMDDEdge(-2.0, accumulate(customBrkt.rbegin() + 1, customBrkt.rend(), customBrkt.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    }).uniqueTableKey);
    QMDDEdge customCZ = mathUtils::add(partialCZ1, partialCZ2);
    this->gateQueue.push(QMDDGate(customCZ));

    return;
}

void QuantumCircuit::addDiffuser() {
    this->addAllH();
    this->addAllX();

    vector<QMDDEdge> customI(this->numQubits, identityEdge);
    QMDDEdge partialCZ1 = accumulate(customI.rbegin() + 1, customI.rend(), customI.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    });
    vector<QMDDEdge> customBrkt(this->numQubits, braketZero);
    QMDDEdge partialCZ2 = QMDDEdge(-2.0, accumulate(customBrkt.rbegin() + 1, customBrkt.rend(), customBrkt.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    }).uniqueTableKey);
    QMDDEdge customCZ = mathUtils::add(partialCZ1, partialCZ2);
    this->gateQueue.push(QMDDGate(customCZ));

    this->addAllX();
    this->addAllH();
    return;
}

void QuantumCircuit::reset(int qubitIndex) {

}

void QuantumCircuit::globalPhase(double lamda) {
    QMDDEdge result = QMDDEdge(exp(i * lamda), nullptr);
    this->gateQueue.push(result);
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
    QMDDGate m0 = accumulate(edges0.rbegin() + 1, edges0.rend(), edges0.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    });
    QMDDGate m1 = accumulate(edges1.rbegin() + 1, edges1.rend(), edges1.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    });
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