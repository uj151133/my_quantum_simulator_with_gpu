#include "circuit.hpp"

int QuantumCircuit::getMaxDepth(optional<int> start, optional<int> end) const {
    int maxDepth = 0;
    int rangeStart = start.value_or(0);
    int rangeEnd = end.value_or(this->numQubits - 1);
    for (int i = rangeStart; i <= rangeEnd; ++i) {
        maxDepth = max(maxDepth, static_cast<int>(this->wires[i].size()));
    }
    return maxDepth;
}

void QuantumCircuit::normalizeLayer() {
    int maxDepth = this->getMaxDepth(optional<int>(), optional<int>());
    for (int q = 0; q < numQubits; q++) {
        while (this->wires[q].size() < maxDepth) {
            this->wires[q].push_back({Type::I, gate::I()});
        }
    }
    for (int depth = 0; depth < maxDepth; depth++) {
        vector<Part> parts;
        for (int q = 0; q < numQubits; q++) {
            cout << "depth: " << depth << ", q: " << q << ", type: " << this->wires[q][depth].type << endl;
            parts.push_back(this->wires[q][depth]);
        }
        while (!parts.empty() && parts.back().type == Type::I || parts.back().type == Type::Void) {
            parts.pop_back();
        }
        vector<QMDDEdge> edges;
        while (!parts.empty()) {
            edges.push_back(parts.front().gate.getInitialEdge());
            parts.erase(parts.begin());
        }
        if (!edges.empty()) {
            QMDDGate result = accumulate(edges.begin(), edges.end(), edges[0], mathUtils::kronWrapper);
            this->layer.push(result);
        }
    }
}


QuantumCircuit::QuantumCircuit(int numQubits, QMDDState initialState) : numQubits(numQubits), finalState(initialState) {
    call_once(initExtendedEdgeFlag, initExtendedEdge);
    this->wires.resize(numQubits);
    if (numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }
}

QuantumCircuit::QuantumCircuit(int numQubits) : numQubits(numQubits), finalState(state::Ket0()) {
    this->wires.resize(numQubits);
    call_once(initExtendedEdgeFlag, initExtendedEdge);
    if (numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }

    for (int i = 1; i < numQubits; i++) {
        finalState = mathUtils::kron(finalState.getInitialEdge(), state::Ket0().getInitialEdge());
    }
}

queue<QMDDGate> QuantumCircuit::getLayer() const {
    return this->layer;
}

QMDDState QuantumCircuit::getFinalState() const {
    return this->finalState;
}

void QuantumCircuit::addI(int qubitIndex) {
    return;
}

void QuantumCircuit::addPh(const vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, delta] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::Ph, gate::Ph(delta)});
    }
    return;
}

void QuantumCircuit::addX(const vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::X, gate::X()});
    }
    return;
}

void QuantumCircuit::addAllX() {
    for (int i = 0; i < numQubits; i++) {
        this->wires[i].push_back({Type::X, gate::X()});
    }
    return;
}

void QuantumCircuit::addY(const vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::Y, gate::Y()});
    }
    return;
}

void QuantumCircuit::addZ(const vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::Z, gate::Z()});
    }
    return;
}

void QuantumCircuit::addS(const vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::S, gate::S()});
    }
    return;
}

void QuantumCircuit::addSdg(const vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::Sdg, gate::Sdg()});
    }
    return;
}

void QuantumCircuit::addV(const vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::V, gate::V()});
    }
    return;
}

void QuantumCircuit::addH(const vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::H, gate::H()});
    }
    return;
}

void QuantumCircuit::addAllH() {
    for (int i = 0; i < numQubits; i++) {
        this->wires[i].push_back({Type::H, gate::H()});
    }
    return;
}

void QuantumCircuit::addCX(int controlIndex, int targetIndex) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
    QMDDEdge customCX;
    vector<QMDDEdge> edges;
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

    this->wires[minIndex].push_back({Type::CX, QMDDGate(customCX)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::Void, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addVarCX(int controlIndex, int targetIndex) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
    QMDDEdge customVarCX;
    if(targetIndex - controlIndex == 1) {
        customVarCX = gate::varCX().getInitialEdge();
    }else {
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
        customVarCX = mathUtils::add(partialVarCX[0], partialVarCX[1]);
    }
    this->wires[minIndex].push_back({Type::varCX, QMDDGate(customVarCX)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::Void, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addCZ(int controlIndex, int targetIndex) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
    QMDDEdge customCZ;
    if(targetIndex - controlIndex == 1) {
        customCZ = gate::CZ().getInitialEdge();
    }else {
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
        customCZ = mathUtils::add(partialCZ[0], partialCZ[1]);
    }
    this->wires[minIndex].push_back({Type::CZ, QMDDGate(customCZ)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::Void, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addSWAP(int qubitIndex1, int qubitIndex2) {
    int minIndex = min(qubitIndex1, qubitIndex2);
    int maxIndex = max(qubitIndex1, qubitIndex2);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
    QMDDEdge customSWAP;
    if(abs(qubitIndex2 - qubitIndex1) == 1) {
        customSWAP = gate::SWAP().getInitialEdge();
    }else {
        size_t numIndex =  maxIndex - minIndex + 1;
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
    this->wires[minIndex].push_back({Type::SWAP, QMDDGate(customSWAP)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::Void, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addP(const vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, phi] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::Ph, gate::P(phi)});
    }
    return;
}

void QuantumCircuit::addT(const vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::T, gate::T()});
    }
    return;
}

void QuantumCircuit::addTdg(const vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::Tdg, gate::Tdg()});
    }
    return;
}

void QuantumCircuit::addCP(int controlIndex, int targetIndex, double phi) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
    QMDDEdge customCP;
    if(targetIndex - controlIndex == 1) {
        customCP = gate::CP(phi).getInitialEdge();
    }else {
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
        customCP = mathUtils::add(partialCP[0], partialCP[1]);
    }
    this->wires[minIndex].push_back({Type::CP, QMDDGate(customCP)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::Void, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addCS(int controlIndex, int targetIndex) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
    QMDDEdge customCS;
    if(targetIndex - controlIndex == 1) {
        customCS = gate::CS().getInitialEdge();
    }else if(controlIndex < targetIndex) {
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
        customCS = mathUtils::add(partialCS[0], partialCS[1]);
    }
    this->wires[minIndex].push_back({Type::CS, QMDDGate(customCS)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::Void, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addR(const vector<pair<int, pair<double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [theta, phi] = params;
        this->wires[qubitIndex].push_back({Type::R, gate::R(theta, phi)});
    }
    return;
}

void QuantumCircuit::addRx(const vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::Rx, gate::Rx(theta)});
    }
    return;
}

void QuantumCircuit::addRy(const vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::Ry, gate::Ry(theta)});
    }
    return;
}

void QuantumCircuit::addRz(const vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::Rz, gate::Rz(theta)});
    }
    return;
}

void QuantumCircuit::addU(const vector<pair<int, tuple<double, double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [theta, phi, lambda] = params;
        this->wires[qubitIndex].push_back({Type::U, gate::U(theta, phi, lambda)});
    }
    return;
}

void QuantumCircuit::addU1(const vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::U1, gate::U1(theta)});
    }
    return;
}

void QuantumCircuit::addU2(const vector<pair<int, pair<double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [phi, lambda] = params;
        this->wires[qubitIndex].push_back({Type::U2, gate::U2(phi, lambda)});
    }
    return;
}

void QuantumCircuit::addU3(const vector<pair<int, tuple<double, double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [theta, phi, lambda] = params;
        this->wires[qubitIndex].push_back({Type::U3, gate::U3(theta, phi, lambda)});
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
        // gateQueue.push(result);
        return;
    }
}

void QuantumCircuit::addGate(int qubitIndex, const QMDDGate& gate) {
    this->wires[qubitIndex].push_back({Type::Other, gate});
    return;
}

void QuantumCircuit::addOracle(int omega) {
    size_t numIndex;
    if (omega == 0) {
        numIndex = 1;
    } else {
        numIndex = static_cast<size_t>(log2(omega)) + 1;
    }

    vector<QMDDEdge> customI(numIndex, identityEdge);
    QMDDEdge partialOracle1 = accumulate(customI.begin() + 1, customI.end(), customI[0], mathUtils::kronForDiagonal);
    vector<QMDDEdge> customCZ;
    for (int bitPosition = 0; bitPosition < numIndex; ++bitPosition) {
        int bitValue = (omega >> bitPosition) & 1;
        if (bitValue == 0) {
            customCZ.push_back(braketZero);
        } else {
            customCZ.push_back(braketOne);
        }
    }

    QMDDEdge partialOracle2 = QMDDEdge(-2.0, accumulate(customCZ.begin() + 1, customCZ.end(), customCZ[0], mathUtils::kronWrapper).uniqueTableKey);
    QMDDEdge customOracle = mathUtils::add(partialOracle1, partialOracle2);
    // gateQueue.push(QMDDGate(customOracle));
    return;
}

void QuantumCircuit::addIAM() {
    this->addAllH();

    vector<QMDDEdge> customCZ(numQubits, braketZero);
    QMDDEdge partialIAM1 = QMDDEdge(2.0, accumulate(customCZ.begin() + 1, customCZ.end(), customCZ[0], mathUtils::kronWrapper).uniqueTableKey);
    vector<QMDDEdge> customI(numQubits, identityEdge);
    QMDDEdge partialIAM2 = QMDDEdge(-1.0, accumulate(customI.begin() + 1, customI.end(), customI[0], mathUtils::kronWrapper).uniqueTableKey);
    QMDDEdge customIAM = mathUtils::add(partialIAM1, partialIAM2);
    // gateQueue.push(QMDDGate(customIAM));

    this->addAllH();
    return;
}

void QuantumCircuit::addBarrier() {
    int maxDepth = this->getMaxDepth(optional<int>(), optional<int>());
    for (int q = 0; q < numQubits; q++) {
        while (this->wires[q].size() < maxDepth) {
            this->wires[q].push_back({Type::I, gate::I()});
        }
    }
    for (int i = 0; i < numQubits; i++) {
        this->wires[i].push_back({Type::Void, QMDDGate()});
    }
    return;
}

void QuantumCircuit::simulate() {
    this->normalizeLayer();
    int i = 0;
    while (!this->layer.empty()) {
        cout << "number of gates: " << i++ << endl;
        QMDDGate currentGate = this->layer.front();
        cout << "Current gate: " << currentGate << endl;
        cout << "Current state: " << finalState << endl;

        cout << "============================================================\n" << endl;
        this->layer.pop();
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