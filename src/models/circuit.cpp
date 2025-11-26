#include "circuit.hpp"

extern "C" void record_time(void (*cb)(), double* elapsed_ms);
thread_local QuantumCircuit* g_tls_qc = nullptr;

extern "C" void qc_critical_block() {
    g_tls_qc->criticalExecute();
}

QuantumCircuit::QuantumCircuit(int numQubits, QMDDState initialState) : numQubits(numQubits), finalState(initialState) {
    call_once(initExtendedEdgeFlag, initExtendedEdge);
    this->wires.resize(this->numQubits);
    if (this->numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }
    this->quantumRegister.resize(1);
    this->setRegister(0, this->numQubits);
    this->swapTable_.resize(this->numQubits);
    this->phy2log_.resize(this->numQubits);
    this->log2phy_.resize(this->numQubits);
    iota(this->swapTable_.begin(), this->swapTable_.end(), 0);
    iota(this->phy2log_.begin(), this->phy2log_.end(), 0);
    iota(this->log2phy_.begin(), this->log2phy_.end(), 0);
}

QuantumCircuit::QuantumCircuit(int numQubits) : numQubits(numQubits), finalState(state::Ket0()) {
    this->wires.resize(this->numQubits);
    call_once(initExtendedEdgeFlag, initExtendedEdge);
    if (this->numQubits < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }

    for (int i = 1; i < this->numQubits; i++) {
        this->finalState = mathUtils::kron(state::Ket0().getInitialEdge(), this->finalState.getInitialEdge());
    }
    this->quantumRegister.resize(1);
    this->setRegister(0, this->numQubits);
    this->swapTable_.resize(this->numQubits);
    this->phy2log_.resize(this->numQubits);
    this->log2phy_.resize(this->numQubits);
    iota(this->swapTable_.begin(), this->swapTable_.end(), 0);
    iota(this->phy2log_.begin(), this->phy2log_.end(), 0);
    iota(this->log2phy_.begin(), this->log2phy_.end(), 0);
}

static inline bool isCancer(const string& gU) {
    return (gU == "H" || gU == "RX" || gU == "RY" || gU == "V" || gU == "VDG");
}

vector<int> QuantumCircuit::countCancer() const {
    vector<int> score(this->numQubits, 0);
    for (const auto& o : this->irLog_) {
        const string g = Core::upper(o.tag);
        if (o.qubits.size() == 1) {
            if (isCancer(g)) {
                score[o.qubits[0]] += 4;
            }
        } else if (o.qubits.size() >= 2) {
            for (size_t i = 0; i < o.qubits.size(); ++i) {
                if (i == o.qubits.size() - 1  && o.tag != "CZ") {
                    score[o.qubits[i]] += 2;
                }
            }
        }
    }
    cout << "Initial cancer scores: ";
    for (const auto& o : this->irLog_) {
        if (o.qubits.size() == 2) {
            int c = o.qubits[0], t = o.qubits[1];
            int diff = abs(score[c] - score[t]);
            if (diff > 0) {
                cout << "Adjusting scores between qubits " << c << " and " << t << ": " << score[c] << " vs " << score[t] << endl;
                if (score[c] > score[t]) {
                    score[c] -= diff * 0.3;
                    score[t] += diff * 0.3;
                } else {
                    score[c] += diff * 0.3;
                    score[t] -= diff * 0.3;
                }
            }
        }
    }
    return score;
}

void QuantumCircuit::consult() {
    auto score = this->countCancer();

    vector<int> order(this->numQubits);
    iota(order.begin(), order.end(), 0); // logical indices

    stable_sort(order.begin(), order.end(),
        [&](int a, int b){
            if (score[a] != score[b]) return score[a] < score[b]; // 昇順
            return a < b; // タイブレーク安定化
        });

    for (int p = 0; p < this->numQubits; ++p) {
        int l = order[p];
        this->phy2log_[p] = l;
        this->log2phy_[l] = p;
    }
}

void QuantumCircuit::emitIR(const vector<Core>& ops){
    bool saved = this->irEnabled_;
    this->irEnabled_ = false;

    for (auto o : ops){
        o.normalize();
        for (auto& q : o.qubits) q = this->log2phy_[q];

        const string g = Core::upper(o.tag);
        if(g=="H") this->addH(o.qubits.at(0));
        else if(g=="X") this->addX(o.qubits.at(0));
        else if(g=="Y") this->addY(o.qubits.at(0));
        else if(g=="Z") this->addZ(o.qubits.at(0));
        else if(g=="P"||g=="U1") this->addP(o.qubits.at(0), (o.phi!=0.0? o.phi : o.theta));
        else if(g=="RX") this->addRx(o.qubits.at(0), o.theta);
        else if(g=="RY") this->addRy(o.qubits.at(0), o.theta);
        else if(g=="RZ") this->addRz(o.qubits.at(0), o.theta);
        else if(g=="CX"||g=="CNOT") this->addCX(o.qubits.at(0), o.qubits.at(1));
        else if(g=="CZ") this->addCZ(o.qubits.at(0), o.qubits.at(1));
        else if(g=="CP") this->addCP(o.qubits.at(0), o.qubits.at(1), (o.phi!=0.0? o.phi : o.theta));
        else if(g=="CRZ") this->addCP(o.qubits.at(0), o.qubits.at(1), o.theta); // 仮: CPで代用
        else if(g=="SWAP") this->addSWAP(o.qubits.at(0), o.qubits.at(1));
        else if(g=="TOFFOLI"||g=="CCX") this->addToff({o.qubits.at(0), o.qubits.at(1)}, o.qubits.at(2));
        else if(g=="BARRIER") this->addBarrier();
    }

    this->irEnabled_ = saved;
    this->irLog_.clear();
}

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

    this->layer_.clear();
    this->layer_.resize(maxDepth);

    for (int q = 0; q < this->numQubits; q++) {
        while (this->wires[q].size() < maxDepth) {
            this->wires[q].push_back({Type::I, gate::I()});
        }
    }
    for (int depth = 0; depth < maxDepth; depth++) {
        vector<Part> parts;
        for (int q = 0; q < this->numQubits; q++) {
            parts.push_back(this->wires[q][depth]);
        }
        while (!parts.empty() && (parts.back().type == Type::I || parts.back().type == Type::VOID || parts.back().type == Type::BAN || parts.back().type == Type::JOKER)) {
            parts.pop_back();
        }
        while (!parts.empty()) {
            cout << "Processing part of type: " << parts.front().type << " at depth " << depth << endl;
            if (parts.front().type != Type::VOID) {
                this->layer_[depth].push_back(parts.front().gate.getInitialEdge());
            }
            parts.erase(parts.begin());
        }
    }
    this->wires.clear();
    this->wires.resize(this->numQubits);
}

void QuantumCircuit::build() {
    for (auto& layer : this->layer_) {
        if (layer.empty()) continue;
        QMDDEdge result = accumulate(
            layer.rbegin() + 1, layer.rend(), layer.back(),
            [](const QMDDEdge& accumulated, const QMDDEdge& current) {
                return mathUtils::kron(current, accumulated);
            }
        );
        this->gateQueue_.push(QMDDGate(result));
    }
}

void QuantumCircuit::smartInsert(const vector<int>& qubitIndices, const Part& part) {
    int minIndex = *min_element(qubitIndices.begin(), qubitIndices.end());
    int maxIndex = *max_element(qubitIndices.begin(), qubitIndices.end());
    int JOKERDepth = this->searchJOKER(qubitIndices);
    if (CONFIG.circuit.joker && JOKERDepth != -1) {
        this->wires[minIndex][JOKERDepth] = part;
        for (int i = minIndex + 1; i <= maxIndex; ++i) {
            this->wires[i][JOKERDepth] = {Type::VOID, QMDDGate()};
        }
    } else {
        this->wires[minIndex].push_back(part);
        for (int i = minIndex + 1; i <= maxIndex; ++i) {
            this->wires[i].push_back({Type::VOID, QMDDGate()});
        }
    }

    return;
}

int QuantumCircuit::searchJOKER(const vector<int>& qubitIndices) {
    if (qubitIndices.empty()) return -1;
    int minIndex = *min_element(qubitIndices.begin(), qubitIndices.end());
    int maxIndex = *max_element(qubitIndices.begin(), qubitIndices.end());
    int maxWireSize = 0;
    for (int i = minIndex; i <= maxIndex; ++i) {
        maxWireSize = max(maxWireSize, static_cast<int>(this->wires[i].size()));
    }
    int JOKERDepth = -1;
    for (int depth = maxWireSize - 1; depth >= 0; --depth) {
        bool allGreen = true;
        for (int i = minIndex; i <= maxIndex; ++i) {
            auto t = this->wires[i][depth].type;
            if (t != Type::JOKER && t != Type::I) {
                if (t != Type::BAN && t != Type::VOID) {
                    return JOKERDepth;
                } else {
                    allGreen = false;
                    break;
                }
            }
        }
        if (allGreen) {
            JOKERDepth = depth;
        }
    }
    return JOKERDepth;
}


queue<QMDDGate> QuantumCircuit::getGateQueue() const {
    return this->gateQueue_;
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

void QuantumCircuit::addPh(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, delta] : qubitParams) {
        this->smartInsert({qubitIndex}, {Type::Ph, gate::Ph(delta)});
    }
    return;
}

void QuantumCircuit::addPh(int qubitIndex, double delta) {
    this->smartInsert({qubitIndex}, {Type::Ph, gate::Ph(delta)});
    return;
}

void QuantumCircuit::addX(int qubitIndex) {
    if (this->irEnabled_) { this->irLog_.push_back(Core{.tag="X", .qubits={this->resolveQubit(qubitIndex)}}); return; }
    this->smartInsert({qubitIndex}, {Type::X, gate::X()});
    return;
}

void QuantumCircuit::addX(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->smartInsert({qubitIndex}, {Type::X, gate::X()});
    }
    return;
}

void QuantumCircuit::addAllX() {
    for (int i = 0; i < numQubits; i++) {
        this->smartInsert({i}, {Type::X, gate::X()});
    }
    return;
}

void QuantumCircuit::addY(int qubitIndex) {
    this->smartInsert({qubitIndex}, {Type::Y, gate::Y()});
    return;
}

void QuantumCircuit::addY(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->smartInsert({qubitIndex}, {Type::Y, gate::Y()});
    }
    return;
}

void QuantumCircuit::addZ(int qubitIndex) {
    if(this->irEnabled_){
        Core c; c.tag="Z"; c.qubits={resolveQubit(qubitIndex)}; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
    this->smartInsert({qubitIndex}, {Type::Z, gate::Z()});
    return;
}

void QuantumCircuit::addZ(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->smartInsert({qubitIndex}, {Type::Z, gate::Z()});
    }
    return;
}

void QuantumCircuit::addS(int qubitIndex) {
    this->smartInsert({qubitIndex}, {Type::S, gate::S()});
    return;
}

void QuantumCircuit::addS(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->smartInsert({qubitIndex}, {Type::S, gate::S()});
    }
    return;
}

void QuantumCircuit::addSdg(int qubitIndex) {
    this->smartInsert({qubitIndex}, {Type::Sdg, gate::Sdg()});
    return;
}

void QuantumCircuit::addSdg(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->smartInsert({qubitIndex}, {Type::Sdg, gate::Sdg()});
    }
    return;
}

void QuantumCircuit::addV(int qubitIndex) {
    if(this->irEnabled_){
        Core c; c.tag="V"; c.qubits={resolveQubit(qubitIndex)}; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }

    if (CONFIG.circuit.joker) {
        int maxDepth = this->getMaxDepth(qubitIndex, this->numQubits - 1);

        while (this->wires[qubitIndex].size() < maxDepth) {
            this->wires[qubitIndex].push_back({Type::I, gate::I()});
        }
        for (int index = qubitIndex + 1; index < this->numQubits; index++) {
            while (this->wires[index].size() < maxDepth) {
                this->wires[index].push_back({Type::JOKER, gate::I()});
            }
        }
        this->wires[qubitIndex].push_back({Type::V, gate::V()});
        for (int index = qubitIndex + 1; index < this->numQubits; index++) {
            this->wires[index].push_back({Type::BAN, QMDDGate()});
        }
    } else {
        this->wires[qubitIndex].push_back({Type::V, gate::V()});
    }
    return;
}

void QuantumCircuit::addH(int qubitIndex) {
    if(this->irEnabled_){
        Core c; c.tag="H"; c.qubits={resolveQubit(qubitIndex)}; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
    if (CONFIG.circuit.joker) {
        int maxDepth = this->getMaxDepth(qubitIndex, this->numQubits - 1);

        while (this->wires[qubitIndex].size() < maxDepth) {
            this->wires[qubitIndex].push_back({Type::I, gate::I()});
        }
        for (int index = qubitIndex + 1; index < this->numQubits; index++) {
            while (this->wires[index].size() < maxDepth) {
                this->wires[index].push_back({Type::JOKER, gate::I()});
            }
        }
        this->wires[qubitIndex].push_back({Type::H, gate::H()});
        for (int index = qubitIndex + 1; index < this->numQubits; index++) {
            this->wires[index].push_back({Type::BAN, QMDDGate()});
        }
    } else {
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
    if(this->irEnabled_){
        Core c; c.tag="CX"; c.qubits={resolveQubit(controlIndex), resolveQubit(targetIndex)}; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
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
        customCX = gate::CX1().getInitialEdge();
    } else {
        array<QMDDEdge, 2> partialCX;
        if (maxIndex == controlIndex) {
            partialCX[0] = braketZero;
            partialCX[1] = braketOne;
        } else {
            partialCX[0] = identityEdge;
            partialCX[1] = gate::X().getInitialEdge();
        }
        for (int index = maxIndex - 1; index >= minIndex; index--) {
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
    this->smartInsert({minIndex, maxIndex}, {Type::CX, QMDDGate(customCX)});
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
    if (maxIndex - minIndex > 1) {
        for (int index = minIndex + 1; index <= maxIndex; index++) {
            this->wires[index].push_back({Type::VOID, QMDDGate()});
        }
    }
    return;
}

void QuantumCircuit::addCY(int controlIndex, int targetIndex) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
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
    this->wires[minIndex].push_back({Type::CY, QMDDGate(customCY)});
    if (maxIndex - minIndex > 1) {
        for (int index = minIndex + 1; index <= maxIndex; index++) {
            this->wires[index].push_back({Type::VOID, QMDDGate()});
        }
    }
    return;
}

void QuantumCircuit::addCZ(int controlIndex, int targetIndex) {
    if(this->irEnabled_){
        Core c; c.tag="CZ"; c.qubits={resolveQubit(controlIndex), resolveQubit(targetIndex)}; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
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
    this->smartInsert({minIndex, maxIndex}, {Type::CZ, QMDDGate(customCZ)});
    return;
}

void QuantumCircuit::addSWAP(int qubitIndex1, int qubitIndex2) {
    swap(this->swapTable_[qubitIndex1], this->swapTable_[qubitIndex2]);
    return;
}

void QuantumCircuit::addP(int qubitIndex, double phi) {
    this->smartInsert({qubitIndex}, {Type::P, gate::P(phi)});
    return;
}

void QuantumCircuit::addP(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, phi] : qubitParams) {
        this->smartInsert({qubitIndex}, {Type::P, gate::P(phi)});
    }
    return;
}

void QuantumCircuit::addT(int qubitIndex) {
    this->smartInsert({qubitIndex}, {Type::T, gate::T()});
    return;
}

void QuantumCircuit::addT(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->smartInsert({qubitIndex}, {Type::T, gate::T()});
    }
    return;
}

void QuantumCircuit::addTdg(int qubitIndex) {
    this->smartInsert({qubitIndex}, {Type::Tdg, gate::Tdg()});
}

void QuantumCircuit::addTdg(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->smartInsert({qubitIndex}, {Type::Tdg, gate::Tdg()});
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

void QuantumCircuit::addCH(int controlIndex, int targetIndex) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
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
    this->wires[minIndex].push_back({Type::CH, QMDDGate(customCH)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addRx(int qubitIndex, double theta) {
    if(this->irEnabled_){
        Core c; c.tag="Rx"; c.qubits={resolveQubit(qubitIndex)}; c.theta=theta; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
    if (CONFIG.circuit.joker) {
        int maxDepth = this->getMaxDepth(qubitIndex, this->numQubits - 1);

        while (this->wires[qubitIndex].size() < maxDepth) {
            this->wires[qubitIndex].push_back({Type::I, gate::I()});
        }
        for (int index = qubitIndex + 1; index < this->numQubits; index++) {
            while (this->wires[index].size() < maxDepth) {
                this->wires[index].push_back({Type::JOKER, gate::I()});
            }
        }
        this->wires[qubitIndex].push_back({Type::Rx, gate::Rx(theta)});
        for (int index = qubitIndex + 1; index < this->numQubits; index++) {
            this->wires[index].push_back({Type::BAN, QMDDGate()});
        }
    } else {
        this->wires[qubitIndex].push_back({Type::Rx, gate::Rx(theta)});
    }
    return;
}

void QuantumCircuit::addRy(int qubitIndex, double theta) {
    if(this->irEnabled_){
        Core c; c.tag="Ry"; c.qubits={resolveQubit(qubitIndex)}; c.theta=theta; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
    if (CONFIG.circuit.joker) {
        int maxDepth = this->getMaxDepth(qubitIndex, this->numQubits - 1);

        while (this->wires[qubitIndex].size() < maxDepth) {
            this->wires[qubitIndex].push_back({Type::I, gate::I()});
        }
        for (int index = qubitIndex + 1; index < this->numQubits; index++) {
            while (this->wires[index].size() < maxDepth) {
                this->wires[index].push_back({Type::JOKER, gate::I()});
            }
        }
        this->wires[qubitIndex].push_back({Type::Ry, gate::Ry(theta)});
        for (int index = qubitIndex + 1; index < this->numQubits; index++) {
            this->wires[index].push_back({Type::BAN, QMDDGate()});
        }
    } else {
        this->wires[qubitIndex].push_back({Type::Ry, gate::Ry(theta)});
    }
    return;
}

void QuantumCircuit::addRz(int qubitIndex, double theta) {
    if(this->irEnabled_){
        Core c; c.tag="RZ"; c.qubits={resolveQubit(qubitIndex)}; c.theta=theta; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
    this->smartInsert({qubitIndex}, {Type::Rz, gate::Rz(theta)});
    return;
}

void QuantumCircuit::addRz(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->smartInsert({qubitIndex}, {Type::Rz, gate::Rz(theta)});
    }
    return;
}

void QuantumCircuit::addFREDKIN(int controlIndex, int targetIndex1, int targetIndex2) {
    if (numQubits < 3) {
        throw invalid_argument("Cannot add Fredkin gate to less than 3 qubit circuit.");
    }else if (controlIndex == targetIndex1 || controlIndex == targetIndex2 || targetIndex1 == targetIndex2) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(numQubits == 3 && ((controlIndex == 0 && targetIndex1 == 1 && targetIndex2 == 2) || (controlIndex == 0 && targetIndex1 == 2 && targetIndex2 == 1))) {
        // this->gateQueue.push(gate::FREDKIN());
        // TODO: add wire method
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

void QuantumCircuit::addCRx(int controlIndex, int targetIndex, double theta) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
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
    this->wires[minIndex].push_back({Type::CRx, QMDDGate(customCRx)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addCRy(int controlIndex, int targetIndex, double theta) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
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
    this->wires[minIndex].push_back({Type::CRy, QMDDGate(customCRy)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addCRz(int controlIndex, int targetIndex, double theta) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
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
    this->wires[minIndex].push_back({Type::CRz, QMDDGate(customCRz)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addCU(int controlIndex, int targetIndex, double theta, double phi, double lambda) {
    int minIndex = min(controlIndex, targetIndex);
    int maxIndex = max(controlIndex, targetIndex);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
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
    this->wires[minIndex].push_back({Type::CU, QMDDGate(customCU)});
    for (int index = minIndex + 1; index <= maxIndex; index++) {
        this->wires[index].push_back({Type::VOID, QMDDGate()});
    }
    return;
}

void QuantumCircuit::addToff(const array<int, 2>& controlIndexes, int targetIndex) {
    if(this->irEnabled_){
        Core c; c.tag="TOFFOLI"; c.qubits={resolveQubit(controlIndexes[0]), resolveQubit(controlIndexes[1]), resolveQubit(targetIndex)}; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
    int minIndex = min({controlIndexes[0], controlIndexes[1], targetIndex});
    int maxIndex = max({controlIndexes[0], controlIndexes[1], targetIndex});

    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
    if (controlIndexes.size() == 0) {
        throw invalid_argument("Control indexes must not be empty.");
    }else if (numQubits < controlIndexes.size() + 1) {
        throw invalid_argument("Number of control indexes must be at most number of qubits - 1.");
    }else if (controlIndexes.size() == 1) {
        addCX(controlIndexes[0], targetIndex);
    }else {
        array<int, 2> sortedControlIndexes = sorted(controlIndexes);
        // int minIndex = min(*min_element(sortedControlIndexes.begin(), sortedControlIndexes.end()), targetIndex);
        // int maxIndex = max(*max_element(sortedControlIndexes.begin(), sortedControlIndexes.end()), targetIndex);
        // vector<QMDDEdge> edges(minIndex, identityEdge);
        // vector<QMDDEdge> edges;
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
        this->smartInsert({minIndex, maxIndex}, {Type::Toff, QMDDGate(customToff)});
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
    if(this->irEnabled_){
        Core c; c.tag="BARRIER"; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
    int maxDepth = this->getMaxDepth(optional<int>(), optional<int>());
    for (int q = 0; q < this->numQubits; q++) {
        while (this->wires[q].size() < maxDepth) {
            this->wires[q].push_back({Type::I, gate::I()});
        }
    }
    for (int i = 0; i < this->numQubits; i++) {
        this->wires[i].push_back({Type::VOID, QMDDGate()});
    }
    return;
}
void QuantumCircuit::reset(int qubitIndex) {

}

void QuantumCircuit::globalPhase(double lamda) {
    // QMDDEdge result = QMDDEdge(exp(i * lamda), nullptr);
    // this->gateQueue.push(result);
    return;
}

void QuantumCircuit::criticalExecute() {
    this->build();
    int i = 0;
    while (!this->gateQueue_.empty()) {
        cout << "number of gates: " << i++ << endl;
        QMDDGate currentGate = this->gateQueue_.front();
        cout << "Current gate: " << currentGate << endl;
        cout << "Current state: " << this->finalState << endl;

        cout << "============================================================\n" << endl;
        this->gateQueue_.pop();
        this->finalState = QMDDState(mathUtils::mul(currentGate.getInitialEdge(), this->finalState.getInitialEdge()));
    }
    cout << "Final state: " << this->finalState << endl;
    return;
}

void QuantumCircuit::simulate() {
    if (this->irEnabled_) {
        if (CONFIG.circuit.shuffle) {
            this->consult();
        }
        this->emitIR(this->irLog_);
    }
    this->normalizeLayer();
    g_tls_qc = this;
    double elapsed = 0.0;
    record_time(&qc_critical_block, &elapsed);
    this->totalTimeMs_ += elapsed;
    printf("\033[1;36mTotal execution time: %.6f ms\033[0m\n", this->totalTimeMs_);
    cout << this->layer_.size() << " layers executed." << endl;
    cout << this->wires.size() << " qubits used." << endl;
    return;
}

int QuantumCircuit::measure(int qubitIndex) {
    // if(this->irEnabled_){
    //     Core c; c.tag="MEASURE"; c.qubits= {resolveQubit(qubitIndex)}; c.normalize();
    //     this->irLog_.push_back(std::move(c)); return;
    // }
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