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
        while (!parts.empty() && parts.back().type == Type::I || parts.back().type == Type::VOID) {
            parts.pop_back();
        }
        vector<QMDDEdge> edges;
        while (!parts.empty()) {
            edges.push_back(parts.front().gate.getInitialEdge());
            parts.erase(parts.begin());
        }
        if (!edges.empty()) {
            QMDDGate result = accumulate(edges.rbegin() + 1, edges.rend(), edges.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
                return mathUtils::kron(current, accumulated, 0);
            });
            // QMDDGate result = accumulate(edges.begin(), edges.end(), edges[0], mathUtils::kronWrapper);
            this->layer.push(result);
        }
    }
}


QuantumCircuit::QuantumCircuit(int numQubits, QMDDState initialState) : numQubits(numQubits), finalState(initialState) {
    call_once(initExtendedEdgeFlag, initExtendedEdge);
    this->wires.resize(numQubits);
    if (this->numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }
    // this->quantumRegister.resize(1);
    // this->setRegister(0, this->numQubits);
}

QuantumCircuit::QuantumCircuit(int numQubits) : numQubits(numQubits), finalState(state::Ket0()) {
    this->wires.resize(numQubits);
    call_once(initExtendedEdgeFlag, initExtendedEdge);
    if (this->numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }

    for (int i = 1; i < this->numQubits; i++) {
        this->finalState = mathUtils::kron(state::Ket0().getInitialEdge(), this->finalState.getInitialEdge());
    }
    // this->quantumRegister.resize(1);
    // this->setRegister(0, this->numQubits);
}

queue<QMDDGate> QuantumCircuit::getLayer() const {
    return this->layer;
}

QMDDState QuantumCircuit::getFinalState() const {
    return this->finalState;
}

// void QuantumCircuit::setRegister(int registerIdx, int size) {
//     if (registerIdx < 0) {
//         throw out_of_range("Invalid register index.");
//     }
//     this->quantumRegister[registerIdx].resize(size);
//     iota(this->quantumRegister[registerIdx].begin(), this->quantumRegister[registerIdx].end() + size, registerIdx == 0 ? 0 : this->quantumRegister[registerIdx - 1].back() + 1);
// }

void QuantumCircuit::addI(int qubitIndex) {
    return;
}

void QuantumCircuit::addPh(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, delta] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::Ph, gate::Ph(delta)});
    }
    return;
}

void QuantumCircuit::addPh(int qubitIndex, double delta) {
    this->wires[qubitIndex].push_back({Type::Ph, gate::Ph(delta)});
    return;
}

void QuantumCircuit::addX(int qubitIndex) {
    this->wires[qubitIndex].push_back({Type::X, gate::X()});
    return;
}

void QuantumCircuit::addX(vector<int>& qubitIndices) {
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

void QuantumCircuit::addY(int qubitIndex) {
    this->wires[qubitIndex].push_back({Type::Y, gate::Y()});
    return;
}

void QuantumCircuit::addY(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::Y, gate::Y()});
    }
    return;
}

void QuantumCircuit::addZ(int qubitIndex) {
    this->wires[qubitIndex].push_back({Type::Z, gate::Z()});
    return;
}

void QuantumCircuit::addZ(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::Z, gate::Z()});
    }
    return;
}

void QuantumCircuit::addS(int qubitIndex) {
    this->wires[qubitIndex].push_back({Type::S, gate::S()});
    return;
}

void QuantumCircuit::addS(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::S, gate::S()});
    }
    return;
}

void QuantumCircuit::addSdg(int qubitIndex) {
    this->wires[qubitIndex].push_back({Type::Sdg, gate::Sdg()});
    return;
}

void QuantumCircuit::addSdg(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::Sdg, gate::Sdg()});
    }
    return;
}

void QuantumCircuit::addV(int qubitIndex) {
    this->wires[qubitIndex].push_back({Type::V, gate::V()});
    return;
}

void QuantumCircuit::addV(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::V, gate::V()});
    }
    return;
}

void QuantumCircuit::addH(int qubitIndex) {
    this->wires[qubitIndex].push_back({Type::H, gate::H()});
    return;
}

void QuantumCircuit::addH(vector<int>& qubitIndices) {
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
        if (maxIndex == controlIndex) {
            partialCX[0] = braketZero;
            partialCX[1] = braketOne;
        } else {
            partialCX[0] = identityEdge;
            partialCX[1] = gate::X().getInitialEdge();
        }
        for (int index = maxIndex - 1; index >= minIndex; index--){
            if (index == controlIndex) {
                partialCX[0] = mathUtils::kron(braketZero, partialCX[0]);
                partialCX[1] = mathUtils::kron(braketOne, partialCX[1]);
            } else if (index == targetIndex) {
                partialCX[0] = mathUtils::kron(identityEdge, partialCX[0]);
                partialCX[1] = mathUtils::kron(gate::X().getInitialEdge(), partialCX[1]);
            } else {
                partialCX[0] = mathUtils::kron(identityEdge, partialCX[0]);
                partialCX[1] = mathUtils::kron(identityEdge, partialCX[1]);
            }
        }
        customCX = mathUtils::add(partialCX[0], partialCX[1]);
    }

    this->wires[minIndex].push_back({Type::CX, QMDDGate(customCX)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
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
        if (maxIndex == controlIndex) {
            partialVarCX[0] = braketZero;
            partialVarCX[1] = braketOne;
        } else {
            partialVarCX[0] = gate::X().getInitialEdge();
            partialVarCX[1] = identityEdge;
        }
        for (int index = maxIndex - 1; index >= minIndex; index--){
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
        customVarCX = mathUtils::add(partialVarCX[0], partialVarCX[1]);
    }
    this->wires[minIndex].push_back({Type::varCX, QMDDGate(customVarCX)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
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
        if (maxIndex == controlIndex) {
            partialCZ[0] = braketZero;
            partialCZ[1] = braketOne;
        } else {
            partialCZ[0] = identityEdge;
            partialCZ[1] = gate::Z().getInitialEdge();
        }
        for (int index = maxIndex - 1; index >= minIndex; index--){
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
        customCZ = mathUtils::add(partialCZ[0], partialCZ[1]);
    }
    this->wires[minIndex].push_back({Type::CZ, QMDDGate(customCZ)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addSWAP(int qubitIndex1, int qubitIndex2) {
    int minIndex = min(qubitIndex1, qubitIndex2);
    int maxIndex = max(qubitIndex1, qubitIndex2);
    int maxDepth = this->getMaxDepth(qubitIndex1, qubitIndex2);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
    QMDDEdge customSWAP;
    if(maxIndex - minIndex == 0) {
        cout << "Adding SWAP gate between adjacent qubits " << minIndex << " and " << maxIndex << endl;
        customSWAP= gate::SWAP().getInitialEdge();
    }else {
        size_t numIndex =  maxIndex - minIndex + 1;
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
            partialSWAP[i] = mathUtils::mul(partialPreSWAP[i][0], partialPreSWAP[i][1]);
        }
        customSWAP = accumulate(partialSWAP.begin() + 1, partialSWAP.end(), partialSWAP[0], [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::add(accumulated, current);
        });
    }
    this->wires[minIndex].push_back({Type::SWAP, customSWAP});
    cout << "Added SWAP gate: " << *customSWAP.getStartNode() << endl;
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addP(int qubitIndex, double phi) {
    this->wires[qubitIndex].push_back({Type::P, gate::P(phi)});
    return;
}

void QuantumCircuit::addP(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, phi] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::P, gate::P(phi)});
    }
    return;
}

void QuantumCircuit::addT(int qubitIndex) {
    this->wires[qubitIndex].push_back({Type::T, gate::T()});
    return;
}

void QuantumCircuit::addT(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->wires[qubitIndex].push_back({Type::T, gate::T()});
    }
    return;
}

void QuantumCircuit::addTdg(int qubitIndex) {
    this->wires[qubitIndex].push_back({Type::Tdg, gate::Tdg()});
}

void QuantumCircuit::addTdg(vector<int>& qubitIndices) {
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
        customCP = mathUtils::add(partialCP[0], partialCP[1]);
    }
    this->wires[minIndex].push_back({Type::CP, QMDDGate(customCP)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
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
        customCS = mathUtils::add(partialCS[0], partialCS[1]);
    }
    this->wires[minIndex].push_back({Type::CS, QMDDGate(customCS)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addR(int qubitIndex, double theta, double phi) {
    this->wires[qubitIndex].push_back({Type::R, gate::R(theta, phi)});
    return;
}

void QuantumCircuit::addR(vector<pair<int, pair<double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [theta, phi] = params;
        this->wires[qubitIndex].push_back({Type::R, gate::R(theta, phi)});
    }
    return;
}

void QuantumCircuit::addRx(int qubitIndex, double theta) {
    this->wires[qubitIndex].push_back({Type::Rx, gate::Rx(theta)});
    return;
}

void QuantumCircuit::addRx(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::Rx, gate::Rx(theta)});
    }
    return;
}

void QuantumCircuit::addRy(int qubitIndex, double theta) {
    this->wires[qubitIndex].push_back({Type::Ry, gate::Ry(theta)});
    return;
}

void QuantumCircuit::addRy(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::Ry, gate::Ry(theta)});
    }
    return;
}

void QuantumCircuit::addRz(int qubitIndex, double theta) {
    this->wires[qubitIndex].push_back({Type::Rz, gate::Rz(theta)});
    return;
}

void QuantumCircuit::addRz(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::Rz, gate::Rz(theta)});
    }
    return;
}

void QuantumCircuit::addU(int qubitIndex, double theta, double phi, double lambda) {
    this->wires[qubitIndex].push_back({Type::U, gate::U(theta, phi, lambda)});
    return;
}

void QuantumCircuit::addU(vector<pair<int, tuple<double, double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [theta, phi, lambda] = params;
        this->wires[qubitIndex].push_back({Type::U, gate::U(theta, phi, lambda)});
    }
    return;
}

void QuantumCircuit::addU1(int qubitIndex, double theta) {
    this->wires[qubitIndex].push_back({Type::U1, gate::U1(theta)});
    return;
}

void QuantumCircuit::addU1(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->wires[qubitIndex].push_back({Type::U1, gate::U1(theta)});
    }
    return;
}

void QuantumCircuit::addU2(int qubitIndex, double phi, double lambda) {
    this->wires[qubitIndex].push_back({Type::U2, gate::U2(phi, lambda)});
    return;
}

void QuantumCircuit::addU2(vector<pair<int, pair<double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [phi, lambda] = params;
        this->wires[qubitIndex].push_back({Type::U2, gate::U2(phi, lambda)});
    }
    return;
}

void QuantumCircuit::addU3(int qubitIndex, double theta, double phi, double lambda) {
    this->wires[qubitIndex].push_back({Type::U3, gate::U3(theta, phi, lambda)});
    return;
}

void QuantumCircuit::addU3(vector<pair<int, tuple<double, double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [theta, phi, lambda] = params;
        this->wires[qubitIndex].push_back({Type::U3, gate::U3(theta, phi, lambda)});
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
        // this->gateQueue.push(result);

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
        // this->gateQueue.push(result);

        return;
    }
}

void QuantumCircuit::addGate(int qubitIndex, const QMDDGate& gate) {
    this->wires[qubitIndex].push_back({Type::Other, gate});
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
    // this->gateQueue.push(QMDDGate(customCZ));

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
    // this->gateQueue.push(QMDDGate(customCZ));

    this->addAllX();
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
        this->wires[i].push_back({Type::VOID, QMDDGate()});
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
        cout << "Current state: " << this->finalState << endl;

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