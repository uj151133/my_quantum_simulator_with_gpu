#include "circuit.hpp"

const unordered_set<string> cancer = {"H", "V", "Vdg", "VDG", "Rx", "RX", "Ry", "RY"};

extern "C" void record_time(void (*cb)(), double* elapsed_ms);
thread_local QuantumCircuit* g_tls_qc = nullptr;
thread_local size_t g_tls_gate_num = 0;

extern "C" void qc_critical_block() {
    g_tls_qc->criticalExecute();
}

QuantumCircuit::QuantumCircuit(int numQubits, QMDDState initialState) : numQubits_(numQubits), finalState_(initialState) {
    call_once(initExtendedEdgeFlag, initExtendedEdge);
    this->wires.resize(this->numQubits_);
    if (this->numQubits_ < 1) {
        throw std::invalid_argument("Number of qubits must be at least 1.");
    }
    this->quantumRegister_.resize(1);
    this->setRegister(0, this->numQubits_);
    this->swapTable_.resize(this->numQubits_);
    this->phy2log_.resize(this->numQubits_);
    this->log2phy_.resize(this->numQubits_);
    iota(this->swapTable_.begin(), this->swapTable_.end(), 0);
    iota(this->phy2log_.begin(), this->phy2log_.end(), 0);
    iota(this->log2phy_.begin(), this->log2phy_.end(), 0);
}

QuantumCircuit::QuantumCircuit(int numQubits) : numQubits_(numQubits), finalState_(state::Ket0()) {
    this->wires.resize(this->numQubits_);
    call_once(initExtendedEdgeFlag, initExtendedEdge);
    if (this->numQubits_ < 1) {
        throw invalid_argument("Number of qubits must be at least 1.");
    }

    for (int i = 1; i < this->numQubits_; i++) {
        this->finalState_ = mathUtils::kron(state::Ket0().getInitialEdge(), this->finalState_.getInitialEdge());
    }
    this->quantumRegister_.resize(1);
    this->setRegister(0, this->numQubits_);
    this->swapTable_.resize(this->numQubits_);
    this->phy2log_.resize(this->numQubits_);
    this->log2phy_.resize(this->numQubits_);
    iota(this->swapTable_.begin(), this->swapTable_.end(), 0);
    iota(this->phy2log_.begin(), this->phy2log_.end(), 0);
    iota(this->log2phy_.begin(), this->log2phy_.end(), 0);
}

static inline bool isCancer(const string& type) {
    return (cancer.contains(type));
}

vector<int> QuantumCircuit::countCancer() const {
    vector<int> score(this->numQubits_, 0);
    for (const auto& o : this->irLog_) {
        const string g = Core::upper(o.tag);
        if (o.qubits.size() == 1) {
            if (isCancer(g)) {
                score[o.qubits[0]] += PARAMETER.circuit.dealer.nonzeroWeight;
            }
        } else if (o.qubits.size() >= 2) {
            for (size_t i = 0; i < o.qubits.size(); ++i) {
                if (i == o.qubits.size() - 1  && o.tag != "CZ") {
                    score[o.qubits[i]] += PARAMETER.circuit.dealer.targetBitWeight;
                }
            }
        }
    }
    for (const auto& o : this->irLog_) {
        if (o.qubits.size() == 2) {
            int c = o.qubits[0], t = o.qubits[1];
            int diff = abs(score[c] - score[t]);
            if (diff > 0) {
                if (score[c] > score[t]) {
                    score[c] -= diff * PARAMETER.circuit.dealer.shorteningWeight;
                    score[t] += diff * PARAMETER.circuit.dealer.shorteningWeight;
                } else {
                    score[c] += diff * PARAMETER.circuit.dealer.shorteningWeight;
                    score[t] -= diff * PARAMETER.circuit.dealer.shorteningWeight;
                }
            }
        }
    }
    return score;
}

void QuantumCircuit::consult() {
    auto score = this->countCancer();

    vector<int> order(this->numQubits_);
    iota(order.begin(), order.end(), 0); // logical indices

    stable_sort(order.begin(), order.end(),
        [&](int a, int b){
            if (score[a] != score[b]) return score[a] < score[b]; // 昇順
            return a < b; // タイブレーク安定化
        });
    
    // cout << "Sorted qubit order (logical indices): ";
    // for (int l : order) {
    //     cout << l << " ";
    // }
    // cout << endl;

    for (int p = 0; p < this->numQubits_; ++p) {
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
        else if(g=="S") this->addS(o.qubits.at(0));
        else if(g=="SDG"||g=="Sdg") this->addSdg(o.qubits.at(0));
        else if(g=="T") this->addT(o.qubits.at(0));
        else if(g=="TDG"||g=="Tdg") this->addTdg(o.qubits.at(0));
        else if(g=="V") this->addV(o.qubits.at(0));
        else if(g=="RX") this->addRx(o.qubits.at(0), o.theta);
        else if(g=="RY") this->addRy(o.qubits.at(0), o.theta);
        else if(g=="RZ") this->addRz(o.qubits.at(0), o.theta);
        else if(g=="RZZ") this->addRzz(o.qubits.at(0), o.qubits.at(1), o.phi);
        else if(g=="CX"||g=="CNOT") this->addCX(o.qubits.at(0), o.qubits.at(1));
        else if(g=="CZ") this->addCZ(o.qubits.at(0), o.qubits.at(1));
        else if(g=="CP") this->addCP(o.qubits.at(0), o.qubits.at(1), (o.phi!=0.0? o.phi : o.theta));
        else if(g=="CRZ") this->addCRz(o.qubits.at(0), o.qubits.at(1), o.theta);
        else if(g=="SWAP") this->addSWAP(o.qubits.at(0), o.qubits.at(1));
        else if(g=="TOFFOLI"||g=="CCX") this->addToff({o.qubits.at(0), o.qubits.at(1)}, o.qubits.at(2));
        else if(g=="BARRIER") this->addBarrier();
    }

    this->irEnabled_ = saved;
    this->irLog_.clear();
}

queue<QMDDGate> QuantumCircuit::getGateQueue() const {
    return this->gateQueue_;
}

QMDDState QuantumCircuit::getFinalState() const {
    return this->finalState_;
}

int QuantumCircuit::getMaxDepth(optional<int> start, optional<int> end) const {
    int maxDepth = 0;
    int rangeStart = start.value_or(0);
    int rangeEnd = end.value_or(this->numQubits_ - 1);
    for (int i = rangeStart; i <= rangeEnd; ++i) {
        maxDepth = max(maxDepth, static_cast<int>(this->wires[i].size()));
    }
    return maxDepth;
}

double QuantumCircuit::getTotalTimeMs() const {
    return this->totalTimeMs_;
}

void QuantumCircuit::setRegister(int registerIdx, int size) {
    if (registerIdx < 0) {
        throw out_of_range("Invalid register index.");
    }

    if (registerIdx >= static_cast<int>(this->quantumRegister_.size())) {
        this->quantumRegister_.resize(registerIdx + 1);
    }

    this->quantumRegister_[registerIdx].resize(size);
    iota(this->quantumRegister_[registerIdx].begin(), this->quantumRegister_[registerIdx].end(), registerIdx == 0 ? 0 : this->quantumRegister_[registerIdx - 1].back() + 1);
}

// void QuantumCircuit::setIrLog(const vector<Core>& ops) {
//     this->irLog_.clear();
//     for (const auto& o : ops) {
//         this->irLog_.push_back(o);
//     }
// }

void QuantumCircuit::normalizeLayer() {
    int maxDepth = this->getMaxDepth(optional<int>(), optional<int>());

    this->layer_.clear();
    this->layer_.resize(maxDepth);

    for (int q = 0; q < this->numQubits_; q++) {
        while (this->wires[q].size() < maxDepth) {
            this->wires[q].push_back({Type::I, gate::I()});
        }
    }
    for (int depth = 0; depth < maxDepth; depth++) {
        vector<Part> parts;
        for (int q = 0; q < this->numQubits_; q++) {
            parts.push_back(this->wires[q][depth]);
        }
        while (!parts.empty() && (parts.back().type == Type::I || parts.back().type == Type::VOID || parts.back().type == Type::ANKER || parts.back().type == Type::BAN || parts.back().type == Type::JOKER)) {
            parts.pop_back();
        }
        while (!parts.empty()) {
            // cout << "Processing part of type: " << parts.front().type << " at depth " << depth << endl;
            if (parts.front().type != Type::VOID && parts.front().type != Type::ANKER) {
                this->layer_[depth].push_back(parts.front().gate.getInitialEdge());
            }
            parts.erase(parts.begin());
        }
    }
    this->wires.clear();
    this->wires.resize(this->numQubits_);
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
    if (PARAMETER.circuit.mode == "sparse") {
        for (int i = 0; i < minIndex; ++i) {
            this->wires[i].push_back({Type::I, QMDDGate(identityEdge)});
        }
        this->wires[minIndex].push_back(part);
        for (int i = minIndex + 1; i < maxIndex; ++i) {
            this->wires[i].push_back({Type::VOID, QMDDGate()});
        }
        if (qubitIndices.size() >= 2) {
            this->wires[maxIndex].push_back({Type::ANKER, QMDDGate()});
        }
        for (int i = maxIndex + 1; i < this->numQubits_; ++i) {
            this->wires[i].push_back({Type::BAN, QMDDGate()});
        }
    } else if (PARAMETER.circuit.mode == "moderate" && isCancer(toString(part.type))) {
        this->wires[minIndex].push_back(part);
        for (int i = minIndex + 1; i < maxIndex; ++i) {
            this->wires[i].push_back({Type::VOID, QMDDGate()});
        }
        if (qubitIndices.size() >= 2) {
            this->wires[maxIndex].push_back({Type::ANKER, QMDDGate()});
        }
        for (int i = maxIndex + 1; i < this->numQubits_; ++i) {
            this->wires[i].push_back({Type::BAN, QMDDGate()});
        }
    }else if (PARAMETER.circuit.mode == "moderate" && JOKERDepth != -1) {
        this->wires[minIndex][JOKERDepth] = part;
        for (int i = minIndex + 1; i < maxIndex; ++i) {
            this->wires[i][JOKERDepth] = {Type::VOID, QMDDGate()};
        }
        if (qubitIndices.size() >= 2) {
            this->wires[maxIndex][JOKERDepth] = {Type::ANKER, QMDDGate()};
        }
    } else {
        this->wires[minIndex].push_back(part);
        for (int i = minIndex + 1; i < maxIndex; ++i) {
            this->wires[i].push_back({Type::VOID, QMDDGate()});
        }
        if (qubitIndices.size() >= 2) {
            this->wires[maxIndex].push_back({Type::ANKER, QMDDGate()});
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

string QuantumCircuit::upper(string s){ for(auto& c:s) c=(char)toupper((unsigned char)c); return s; }
bool QuantumCircuit::isDiagTag(const string& t){ return Core::isDiagTag(t); }
void QuantumCircuit::enableIR(bool on){
    this->irEnabled_ = on;
    if(on) { while(!this->gateQueue_.empty()) this->gateQueue_.pop(); }
}
void QuantumCircuit::clearIR(){this->irLog_.clear(); }

// void QuantumCircuit::compileIRWithLaw(const law::Options& opt){
//     if(!this->irEnabled_) return;
//     auto best = law::optimize(this->irLog_, opt);
//     while(!this->gateQueue_.empty()) this->gateQueue_.pop();
//     this->emitIR(best);
// }

void QuantumCircuit::preprocess(const law::Options& opt, const string& modelPath){
    if (!this->irEnabled_ || this->irLog_.empty()) return;

    // 1) law を適用
    vector<Core> after_law = law::optimize(this->irLog_, opt);

    // 2) モデルで順序提案（perm）
    aiinfer::SchedulerONNX sched(modelPath);
    auto perm = sched.predict(after_law);
    if (perm.size() != after_law.size())
        throw std::runtime_error("scheduler perm size mismatch");
    
    // 2.5) モデル順位スコア + ヒューリスティック（-cost）を合算
    const size_t N = after_law.size();
    vector<int> rank(N, 0);
    for (size_t pos = 0; pos < N; ++pos) {
        int idx = perm[pos];
        if (idx < 0 || (size_t)idx >= N) throw runtime_error("scheduler perm out of range");
        rank[(size_t)idx] = (int)pos;
    }
    const float denom = (N > 1) ? float(N - 1) : 1.0f;
    auto costs = heauristicCosts(after_law);
    vector<float> combined(N, 0.0f);
    for (size_t i = 0; i < N; ++i) {
        float model_score = (N > 1) ? (float(N - 1 - rank[i]) / denom) : 1.0f; // 高いほど良
        float heur_score  = heauristicScore(costs[i]);                           // = -cost
        combined[i] = model_score + heur_score;
    }
    // 合算スコア降順の希望順を作成
    vector<int> pref(N); iota(pref.begin(), pref.end(), 0);
    stable_sort(pref.begin(), pref.end(),
        [&](int a, int b){ return combined[(size_t)a] > combined[(size_t)b]; });

    // 3) DAG による合法化（合算順をトポ順へ投影）
    auto order = dag::tuneDAG(after_law, pref);
    if (order.size() != after_law.size())
        throw runtime_error("failed to legalize order by DAG");

    // 4) 並べ替え
    vector<Core> reordered(after_law.size());
    for(size_t i=0;i<order.size();++i){
        int idx = order[i];
        if(idx<0 || (size_t)idx>=after_law.size())
            throw runtime_error("legalized index out of range");
        reordered[i] = after_law[(size_t)idx];
    }

    // 5) ゲート化（law はすでに適用済みなのでそのまま出力）
    this->irLog_ = std::move(reordered);
}

void QuantumCircuit::scheduleIRWithModel(const string& modelPath){
    if(this->irLog_.empty()) return;
    // importer を使って順列を取得（既存: 非合法化）
    aiinfer::SchedulerONNX sched(modelPath);
    auto perm = sched.predict(this->irLog_);
    if(perm.size() != this->irLog_.size()) throw runtime_error("scheduler perm size mismatch");
    vector<Core> reordered(this->irLog_.size());
    for(size_t i=0;i<perm.size();++i){
        int idx = perm[i];
        if(idx<0 || (size_t)idx>=this->irLog_.size()) throw runtime_error("scheduler perm out of range");
        reordered[i] = this->irLog_[(size_t)idx];
    }
    this->irLog_.swap(reordered);
}

void QuantumCircuit::buildMetaFromIR(const vector<Core>& ops){
    this->metaQueue_.clear(); this->metaQueue_.reserve(ops.size());
    for(auto x : ops){
        Core d = x; d.normalize();
        d.handle = 0;
        d.edge_nodes = 0;
        this->metaQueue_.push_back(std::move(d));
    }
}

void QuantumCircuit::moveQueueToPending(){
    this->pending_.clear(); this->execIdx_ = 0;
    auto q = this->gateQueue_; // コピー
    while(!q.empty()){ this->pending_.push_back(q.front()); q.pop(); }
    if (this->metaQueue_.size() != this->pending_.size()){
        this->metaQueue_.clear();
        this->metaQueue_.resize(this->pending_.size());
        for (auto& d : this->metaQueue_) { d.tag="FUSED"; d.normalize(); d.handle=0; d.edge_nodes=0; }
    }
    // edge_nodes を埋めたい場合はここで pending_[i].getInitialEdge() から取得
}

vector<Core> QuantumCircuit::snapshotQueueWindow(size_t max_items) const{
    vector<Core> out;
    if (this->execIdx_ >= this->metaQueue_.size()) return out;
    size_t end = min(this->metaQueue_.size(), this->execIdx_ + max_items);
    out.reserve(end - this->execIdx_);
    for (size_t i=this->execIdx_; i<end; ++i) out.push_back(this->metaQueue_[i]);
    return out;
}

void QuantumCircuit::fuseRanges(const vector<pair<int,int>>& ranges){
    if (this->execIdx_ >= this->pending_.size()) return;
    vector<pair<size_t,size_t>> norm;
    for (auto [s,e] : ranges){
        if (s<0 || e<0) continue;
        size_t ss = this->execIdx_ + (size_t)s, ee = this->execIdx_ + (size_t)e;
        if (ss >= this->pending_.size()) continue;
        if (ee >= this->pending_.size()) ee = this->pending_.size()-1;
        if (ss > ee) std::swap(ss, ee);
        if (ss == ee) continue;
        norm.push_back({ss, ee});
    }
    if (norm.empty()) return;
    sort(norm.begin(), norm.end());
    vector<pair<size_t,size_t>> disjoint;
    for (auto seg : norm){
        if (disjoint.empty() || seg.first > disjoint.back().second) disjoint.push_back(seg);
        else disjoint.back().second = max(disjoint.back().second, seg.second);
    }
    for (int i=(int)disjoint.size()-1; i>=0; --i){
        size_t s = disjoint[i].first, e = disjoint[i].second;
        QMDDEdge acc = this->pending_[s].getInitialEdge();
        for (size_t k=s+1; k<=e; ++k){
            // acc = mathUtils::mul(this->pending_[k].getInitialEdge(), acc);
            acc = threadPool.submitFiber([&]() { return mathUtils::mul(this->pending_[k].getInitialEdge(), acc); }).get();
        }
        QMDDGate fused(acc);
        Core d; d.tag="FUSED"; d.normalize(); d.handle=nextFusedId_++;
        // qubits 集合
        vector<int> tmp;
        for(size_t k=s; k<=e; ++k){
            const auto& qs = this->metaQueue_[k].qubits;
            tmp.insert(tmp.end(), qs.begin(), qs.end());
        }
        sort(tmp.begin(), tmp.end());
        tmp.erase(unique(tmp.begin(), tmp.end()), tmp.end());
        d.qubits = std::move(tmp);
        bool all_diag = true;
        for(size_t k=s; k<=e; ++k){
            if (Core::upper(this->metaQueue_[k].shape) != Core::kShapeDiag) { all_diag=false; break; }
        }
        d.shape = all_diag ? Core::kShapeDiag : Core::kShapeFused;
        // d.edge_nodes = fused.getInitialEdge().getNodeCount(); // 実装があれば設定

        this->fusedStore_[d.handle] = fused;

        this->pending_.erase(this->pending_.begin()+s, this->pending_.begin()+e+1);
        this->metaQueue_.erase(this->metaQueue_.begin()+s, this->metaQueue_.begin()+e+1);
        this->pending_.insert(this->pending_.begin()+s, fused);
        this->metaQueue_.insert(this->metaQueue_.begin()+s, std::move(d));
    }
}

void QuantumCircuit::addI(int qubitIndex) {
    return;
}

void QuantumCircuit::addPh(int qubitIndex, double delta) {
    this->smartInsert({qubitIndex}, {Type::Ph, gate::Ph(delta)});
    return;
}

void QuantumCircuit::addPh(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, delta] : qubitParams) {
        this->addPh(qubitIndex, delta);
    }
    return;
}

void QuantumCircuit::addX(int qubitIndex) {
    if (this->irEnabled_) { this->irLog_.push_back(Core{.tag="X", .qubits={this->resolveQubit(qubitIndex)}}); return; }
    this->smartInsert({qubitIndex}, {Type::X, gate::X()});
    return;
}

void QuantumCircuit::addX(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->addX(qubitIndex);
    }
    return;
}

void QuantumCircuit::addAllX() {
    for (int i = 0; i < numQubits_; i++) {
        this->addX(i);
    }
    return;
}

void QuantumCircuit::addY(int qubitIndex) {
    this->smartInsert({qubitIndex}, {Type::Y, gate::Y()});
    return;
}

void QuantumCircuit::addY(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->addY(qubitIndex);
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
        this->addZ(qubitIndex);
    }
    return;
}

void QuantumCircuit::addS(int qubitIndex) {
    this->smartInsert({qubitIndex}, {Type::S, gate::S()});
    return;
}

void QuantumCircuit::addS(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->addS(qubitIndex);
    }
    return;
}

void QuantumCircuit::addSdg(int qubitIndex) {
    this->smartInsert({qubitIndex}, {Type::Sdg, gate::Sdg()});
    return;
}

void QuantumCircuit::addSdg(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->addSdg(qubitIndex);
    }
    return;
}

void QuantumCircuit::addV(int qubitIndex) {
    if(this->irEnabled_){
        Core c; c.tag="V"; c.qubits={resolveQubit(qubitIndex)}; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }

    if (PARAMETER.circuit.mode == "moderate") {
        int maxDepth = this->getMaxDepth(qubitIndex, this->numQubits_ - 1);

        while (this->wires[qubitIndex].size() < maxDepth) {
            this->wires[qubitIndex].push_back({Type::I, gate::I()});
        }
        for (int index = qubitIndex + 1; index < this->numQubits_; index++) {
            while (this->wires[index].size() < maxDepth) {
                this->wires[index].push_back({Type::JOKER, gate::I()});
            }
        }
    }
    this->smartInsert({qubitIndex}, {Type::V, gate::V()});
    return;
}

void QuantumCircuit::addV(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->addV(qubitIndex);
    }
    return;
}

void QuantumCircuit::addH(int qubitIndex) {
    if(this->irEnabled_){
        this->irLog_.push_back(Core{.tag="H", .qubits={resolveQubit(qubitIndex)}});
        return;
    }
    if (PARAMETER.circuit.mode == "moderate") {
        int maxDepth = this->getMaxDepth(qubitIndex, this->numQubits_ - 1);

        while (this->wires[qubitIndex].size() < maxDepth) {
            this->wires[qubitIndex].push_back({Type::I, gate::I()});
        }
        for (int index = qubitIndex + 1; index < this->numQubits_; index++) {
            while (this->wires[index].size() < maxDepth) {
                this->wires[index].push_back({Type::JOKER, gate::I()});
            }
        }
    }
    this->smartInsert({qubitIndex}, {Type::H, gate::H()});
    return;
}

void QuantumCircuit::addH(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->addH(qubitIndex);
    }
    return;
}

void QuantumCircuit::addAllH() {
    for (size_t i = 0; i < this->numQubits_; i++) {
        this->addH(i);
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
        // customCX = mathUtils::add(partialCX[0], partialCX[1]);
        customCX = threadPool.submitFiber([&]() { return mathUtils::add(partialCX[0], partialCX[1]); }).get();
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
        // customVarCX = mathUtils::add(partialVarCX[0], partialVarCX[1]);
        customVarCX = threadPool.submitFiber([&]() { return mathUtils::add(partialVarCX[0], partialVarCX[1]); }).get();
    }
    this->smartInsert({minIndex, maxIndex}, {Type::varCX, QMDDGate(customVarCX)});
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
    // QMDDEdge customCY = mathUtils::add(partialCY[0], partialCY[1]);
    QMDDEdge customCY = threadPool.submitFiber([&]() { return mathUtils::add(partialCY[0], partialCY[1]); }).get();
    this->smartInsert({minIndex, maxIndex}, {Type::CY, QMDDGate(customCY)});
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
        // customCZ = mathUtils::add(partialCZ[0], partialCZ[1]);
        customCZ = threadPool.submitFiber([&]() { return mathUtils::add(partialCZ[0], partialCZ[1]); }).get();
    }
    this->smartInsert({minIndex, maxIndex}, {Type::CZ, QMDDGate(customCZ)});
    return;
}

void QuantumCircuit::addSWAP(int qubitIndex1, int qubitIndex2) {
    swap(this->swapTable_[qubitIndex1], this->swapTable_[qubitIndex2]);
    return;
}

void QuantumCircuit::addP(int qubitIndex, double phi) {
    if(this->irEnabled_){
        this->irLog_.push_back(Core{.tag = "P", .qubits = {resolveQubit(qubitIndex)}, .phi = phi});
        return;
    }
    this->smartInsert({qubitIndex}, {Type::P, gate::P(phi)});
    return;
}

void QuantumCircuit::addP(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, phi] : qubitParams) {
        this->addP(qubitIndex, phi);
    }
    return;
}

void QuantumCircuit::addT(int qubitIndex) {
    if(this->irEnabled_){
        this->irLog_.push_back(Core{.tag = "T", .qubits = {resolveQubit(qubitIndex)}});
        return;
    }
    this->smartInsert({qubitIndex}, {Type::T, gate::T()});
    return;
}

void QuantumCircuit::addT(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->addT(qubitIndex);
    }
    return;
}

void QuantumCircuit::addTdg(int qubitIndex) {
    if(this->irEnabled_){
        this->irLog_.push_back(Core{.tag = "TDG", .qubits = {resolveQubit(qubitIndex)}});
        return;
    }
    this->smartInsert({qubitIndex}, {Type::Tdg, gate::Tdg()});
    return;
}

void QuantumCircuit::addTdg(vector<int>& qubitIndices) {
    for (int qubitIndex : qubitIndices) {
        this->addTdg(qubitIndex);
    }
    return;
}

void QuantumCircuit::addCP(int controlIndex, int targetIndex, double phi) {
    if(this->irEnabled_){
        Core c; c.tag="CP"; c.qubits={resolveQubit(controlIndex), resolveQubit(targetIndex)}; c.phi=phi; c.normalize();
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
        // customCP = mathUtils::add(partialCP[0], partialCP[1]);
        customCP = threadPool.submitFiber([&]() { return mathUtils::add(partialCP[0], partialCP[1]); }).get();
    }
    this->smartInsert({minIndex, maxIndex}, {Type::CP, QMDDGate(customCP)});
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
        // customCS = mathUtils::add(partialCS[0], partialCS[1]);
        customCS = threadPool.submitFiber([&]() { return mathUtils::add(partialCS[0], partialCS[1]); }).get();
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
        this->addR(qubitIndex, theta, phi);
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
    // QMDDEdge customCH = mathUtils::add(partialCH[0], partialCH[1]);
    QMDDEdge customCH = threadPool.submitFiber([&]() { return mathUtils::add(partialCH[0], partialCH[1]); }).get();
    this->smartInsert({minIndex, maxIndex}, {Type::CH, QMDDGate(customCH)});
    return;
}

void QuantumCircuit::addRx(int qubitIndex, double theta) {
    if(this->irEnabled_){
        Core c; c.tag="Rx"; c.qubits={resolveQubit(qubitIndex)}; c.theta=theta; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
    if (PARAMETER.circuit.mode == "moderate") {
        int maxDepth = this->getMaxDepth(qubitIndex, this->numQubits_ - 1);

        while (this->wires[qubitIndex].size() < maxDepth) {
            this->wires[qubitIndex].push_back({Type::I, gate::I()});
        }
        for (int index = qubitIndex + 1; index < this->numQubits_; index++) {
            while (this->wires[index].size() < maxDepth) {
                this->wires[index].push_back({Type::JOKER, gate::I()});
            }
        }
    //     this->wires[qubitIndex].push_back({Type::Rx, gate::Rx(theta)});
    //     for (int index = qubitIndex + 1; index < this->numQubits_; index++) {
    //         this->wires[index].push_back({Type::BAN, QMDDGate()});
    //     }
    // } else {
    //     this->wires[qubitIndex].push_back({Type::Rx, gate::Rx(theta)});
    }
    this->smartInsert({qubitIndex}, {Type::Rx, gate::Rx(theta)});
    return;
}

void QuantumCircuit::addRy(int qubitIndex, double theta) {
    if(this->irEnabled_){
        Core c; c.tag="Ry"; c.qubits={resolveQubit(qubitIndex)}; c.theta=theta; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
    if (PARAMETER.circuit.mode == "moderate") {
        int maxDepth = this->getMaxDepth(qubitIndex, this->numQubits_ - 1);

        while (this->wires[qubitIndex].size() < maxDepth) {
            this->wires[qubitIndex].push_back({Type::I, gate::I()});
        }
        for (int index = qubitIndex + 1; index < this->numQubits_; index++) {
            while (this->wires[index].size() < maxDepth) {
                this->wires[index].push_back({Type::JOKER, gate::I()});
            }
        }
    //     this->wires[qubitIndex].push_back({Type::Ry, gate::Ry(theta)});
    //     for (int index = qubitIndex + 1; index < this->numQubits_; index++) {
    //         this->wires[index].push_back({Type::BAN, QMDDGate()});
    //     }
    // } else {
    //     this->wires[qubitIndex].push_back({Type::Ry, gate::Ry(theta)});
    }
    this->smartInsert({qubitIndex}, {Type::Ry, gate::Ry(theta)});
    return;
}

void QuantumCircuit::addRz(int qubitIndex, double theta) {
    if(this->irEnabled_){
        this->irLog_.push_back(Core{.tag="RZ", .qubits={resolveQubit(qubitIndex)}, .theta=theta});
        return;
    }
    this->smartInsert({qubitIndex}, {Type::Rz, gate::Rz(theta)});
    return;
}

void QuantumCircuit::addRz(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->addRz(qubitIndex, theta);
    }
    return;
}

void QuantumCircuit::addRzz(int qubitIndex1, int qubitIndex2, double phi) {
    if(this->irEnabled_){
        Core c; c.tag="Rzz"; c.qubits={resolveQubit(qubitIndex1), resolveQubit(qubitIndex2)}; c.phi=phi; c.normalize();
        this->irLog_.push_back(std::move(c)); return;
    }
    int minIndex = min(qubitIndex1, qubitIndex2);
    int maxIndex = max(qubitIndex1, qubitIndex2);
    int maxDepth = this->getMaxDepth(minIndex, maxIndex);
    for (int index = minIndex; index <= maxIndex; index++) {
        while (this->wires[index].size() < maxDepth) {
            this->wires[index].push_back({Type::I, gate::I()});
        }
    }
    QMDDEdge customRzz;
    size_t numIndex =  maxIndex - minIndex + 1;
    if (numIndex == 2){
        customRzz = gate::Rzz(phi).getInitialEdge();
    } else {
        // vector<vector<QMDDEdge>> partialPreRzz(pow(2, numIndex), vector<QMDDEdge>(2, identityEdge));
        vector<QMDDEdge> partialRzz(pow(2, numIndex));
        for (size_t i = 0; i < partialRzz.size(); i++){
            int highestBit = (i >> (numIndex - 1)) & 1;
            int lowestBit = i & 1;
            bool parity = highestBit ^ lowestBit;
            int basedIndex = i;
            // int swappedIndex = (i & ~(1ULL << (numIndex - 1))) | (lowestBit << (numIndex - 1));
            // swappedIndex = (swappedIndex & ~1ULL) | highestBit;
            for ([[maybe_unused]] int _ = numIndex - 1; _ >= 0; _--) {
                // bool msbBased = (basedIndex >> (numIndex - 1)) & 1 ;
                bool lsbBased = basedIndex & 1;
                if (lsbBased){
                    if (_ == numIndex - 1){
                        partialRzz[i] = braketOne;
                    }else {
                        partialRzz[i] = mathUtils::kron(braketOne, partialRzz[i]);
                    }
                }else {
                    if (_ == numIndex - 1){
                        partialRzz[i] = braketZero;
                    }else {
                        partialRzz[i] = mathUtils::kron(braketZero, partialRzz[i]);
                    }
                }
                basedIndex >>= 1;
                // basedIndex <<= 1;
            }
            partialRzz[i].weight = parity ? exp(i * phi / 2) : exp(-i * phi / 2);
        }
        customRzz = accumulate(partialRzz.begin() + 1, partialRzz.end(), partialRzz[0], [](const QMDDEdge& accumulated, const QMDDEdge& current) {
            return mathUtils::add(accumulated, current);
        });
    }
    this->smartInsert({minIndex, maxIndex}, {Type::Rzz, QMDDGate(customRzz)});
    return;
}

void QuantumCircuit::addFREDKIN(int controlIndex, int targetIndex1, int targetIndex2) {
    if (this->numQubits_ < 3) {
        throw invalid_argument("Cannot add Fredkin gate to less than 3 qubit circuit.");
    }else if (controlIndex == targetIndex1 || controlIndex == targetIndex2 || targetIndex1 == targetIndex2) {
        throw invalid_argument("Control and target indices must be different.");
    }else if(this->numQubits_ == 3 && ((controlIndex == 0 && targetIndex1 == 1 && targetIndex2 == 2) || (controlIndex == 0 && targetIndex1 == 2 && targetIndex2 == 1))) {
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
    this->smartInsert({qubitIndex}, {Type::U, gate::U(theta, phi, lambda)});
    return;
}

void QuantumCircuit::addU(vector<pair<int, tuple<double, double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [theta, phi, lambda] = params;
        this->addU(qubitIndex, theta, phi, lambda);
    }
    return;
}

void QuantumCircuit::addU1(int qubitIndex, double theta) {
    this->smartInsert({qubitIndex}, {Type::U1, gate::U1(theta)});
    return;
}

void QuantumCircuit::addU1(vector<pair<int, double>>& qubitParams) {
    for (const auto& [qubitIndex, theta] : qubitParams) {
        this->addU1(qubitIndex, theta);
    }
    return;
}

void QuantumCircuit::addU2(int qubitIndex, double phi, double lambda) {
    this->smartInsert({qubitIndex}, {Type::U2, gate::U2(phi, lambda)});
    return;
}

void QuantumCircuit::addU2(vector<pair<int, pair<double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [phi, lambda] = params;
        this->addU2(qubitIndex, phi, lambda);
    }
    return;
}

void QuantumCircuit::addU3(int qubitIndex, double theta, double phi, double lambda) {
    this->smartInsert({qubitIndex}, {Type::U3, gate::U3(theta, phi, lambda)});
    return;
}

void QuantumCircuit::addU3(vector<pair<int, tuple<double, double, double>>>& qubitParams) {
    for (const auto& [qubitIndex, params] : qubitParams) {
        const auto& [theta, phi, lambda] = params;
        this->addU3(qubitIndex, theta, phi, lambda);
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
    // QMDDEdge customCRx = mathUtils::add(partialCRx[0], partialCRx[1]);
    QMDDEdge customCRx = threadPool.submitFiber([&]() { return mathUtils::add(partialCRx[0], partialCRx[1]); }).get();
    this->smartInsert({minIndex, maxIndex}, {Type::CRx, QMDDGate(customCRx)});
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
    // QMDDEdge customCRy = mathUtils::add(partialCRy[0], partialCRy[1]);
    QMDDEdge customCRy = threadPool.submitFiber([&]() { return mathUtils::add(partialCRy[0], partialCRy[1]); }).get();
    this->smartInsert({minIndex, maxIndex}, {Type::CRy, QMDDGate(customCRy)});
    return;
}

void QuantumCircuit::addCRz(int controlIndex, int targetIndex, double theta) {
    if (this->irEnabled_) {
        this->irLog_.push_back(Core{.tag="CRZ", .qubits={resolveQubit(controlIndex), resolveQubit(targetIndex)}, .theta=theta});
        return;
    }
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
    // QMDDEdge customCRz = mathUtils::add(partialCRz[0], partialCRz[1]);
    QMDDEdge customCRz = threadPool.submitFiber([&]() { return mathUtils::add(partialCRz[0], partialCRz[1]); }).get();
    this->smartInsert({minIndex, maxIndex}, {Type::CRz, QMDDGate(customCRz)});
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
    // QMDDEdge customCU = mathUtils::add(partialCU[0], partialCU[1]);
    QMDDEdge customCU = threadPool.submitFiber([&]() { return mathUtils::add(partialCU[0], partialCU[1]); }).get();
    this->smartInsert({minIndex, maxIndex}, {Type::CU, QMDDGate(customCU)});
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
    }else if (this->numQubits_ < controlIndexes.size() + 1) {
        throw invalid_argument("Number of control indexes must be at most number of qubits - 1.");
    }else if (controlIndexes.size() == 1) {
        this->addCX(controlIndexes[0], targetIndex);
    }else {
        array<int, 2> sortedControlIndexes = sorted(controlIndexes);
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
    }else if (this->numQubits_ < controlIndexes.size() + 1) {
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
    this->smartInsert({qubitIndex}, {Type::Other, gate});
    return;;
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
    this->addQFT(this->numQubits_);
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
    // QMDDEdge customCZ = mathUtils::add(partialCZ1, partialCZ2);
    QMDDEdge customCZ = threadPool.submitFiber([&]() { return mathUtils::add(partialCZ1, partialCZ2); }).get();
    if (PARAMETER.circuit.mode == "sparse") {
        this->gateQueue_.push(QMDDGate(customCZ));
    }

    return;
}

void QuantumCircuit::addDiffuser() {
    this->addAllH();
    this->addAllX();

    vector<QMDDEdge> customI(this->numQubits_, identityEdge);
    QMDDEdge partialCZ1 = accumulate(customI.rbegin() + 1, customI.rend(), customI.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    });
    vector<QMDDEdge> customBrkt(this->numQubits_, braketZero);
    QMDDEdge partialCZ2 = QMDDEdge(-2.0, accumulate(customBrkt.rbegin() + 1, customBrkt.rend(), customBrkt.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    }).uniqueTableKey);
    // QMDDEdge customCZ = mathUtils::add(partialCZ1, partialCZ2);
    QMDDEdge customCZ = threadPool.submitFiber([&]() { return mathUtils::add(partialCZ1, partialCZ2); }).get();

    if (PARAMETER.circuit.mode == "sparse") {
        this->gateQueue_.push(QMDDGate(customCZ));
    }

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
    for (int q = 0; q < this->numQubits_; q++) {
        while (this->wires[q].size() < maxDepth) {
            this->wires[q].push_back({Type::I, gate::I()});
        }
    }
    for (int i = 0; i < this->numQubits_; i++) {
        this->wires[i].push_back({Type::VOID, QMDDGate()});
    }
    return;
}
void QuantumCircuit::reset(int qubitIndex) {

}

void QuantumCircuit::globalPhase(double lamda) {
    if (PARAMETER.circuit.mode == "sparse") {
        this->gateQueue_.push(QMDDEdge(exp(i * lamda), nullptr));
    }
    return;
}


void QuantumCircuit::criticalExecute() {
    // this->build();
    // int i = 0;
    // const size_t gateNum = g_tls_gate_num;
    // while (!this->gateQueue_.empty()) {
    //     QMDDGate currentGate = this->gateQueue_.front();
    //     if (PARAMETER.circuit.verbose) {
    //         cout << "Gate Idx: " << i++ << " / " << gateNum << endl;
    //         cout << "Current gate: " << currentGate << endl;
    //         cout << "Current state: " << this->finalState_ << endl;

    //         cout << "============================================================\n" << endl;
    //     }
    //     this->gateQueue_.pop();
    //     // this->finalState_ = QMDDState(mathUtils::mul(currentGate.getInitialEdge(), this->finalState_.getInitialEdge()));
    //     this->finalState_ = threadPool.submitFiber([&]() { return QMDDState(mathUtils::mul(currentGate.getInitialEdge(), this->finalState_.getInitialEdge())); }).get();
    // }
    // return;

    threadPool.submitFiber([&]() {
        this->build();

        int i = 0;
        const size_t gateNum = g_tls_gate_num;

        while (!this->gateQueue_.empty()) {
            QMDDGate currentGate = this->gateQueue_.front();

            if (PARAMETER.circuit.verbose) {
                cout << "Gate Idx: " << i++ << " / " << gateNum << endl;
                cout << "Current gate: " << currentGate << endl;
                cout << "Current state: " << this->finalState_ << endl;
                cout << "============================================================\n" << endl;
            }

            this->gateQueue_.pop();

            // 逐次依存なのでここは順番通りのまま。
            // 並列化は mul() 内部の fiber 分割に任せる。
            this->finalState_ = QMDDState(
                mathUtils::mul(currentGate.getInitialEdge(), this->finalState_.getInitialEdge())
            );
        }
    }).get();
}


bool QuantumCircuit::simulateStep(){
    if (this->pending_.empty() && !this->gateQueue_.empty()){
        this->moveQueueToPending();
    }
    if (this->execIdx_ >= this->pending_.size()) return false;
    QMDDGate currentGate = this->pending_[this->execIdx_];
    // this->finalState_ = QMDDState(mathUtils::mul(currentGate.getInitialEdge(), this->finalState_.getInitialEdge()));
    this->finalState_ = threadPool.submitFiber([&]() { return QMDDState(mathUtils::mul(currentGate.getInitialEdge(), this->finalState_.getInitialEdge())); }).get();
    this->execIdx_++;
    return this->execIdx_ < this->pending_.size();
}


void QuantumCircuit::simulate() {
    if (this->irEnabled_) {
        if (PARAMETER.circuit.dealer.alive) {
            this->consult();
        }
        law::Options opt = law::optionsFromEnv(law::Options{});
        if (PARAMETER.schedulerAI.alive) {
            this->preprocess(opt, ::SCHEDULER_MODEL_PATH);
        }
        this->emitIR(this->irLog_);
        this->irLog_.clear();
    }
    this->normalizeLayer();

    // if (this->pending_.empty() && !this->gateQueue_.empty()){
    //     this->moveQueueToPending();
    // }

    int gateNum = this->layer_.size();
    g_tls_qc = this;
    g_tls_gate_num = gateNum;
    double elapsed = .0;
    if (gateNum != 0) {
        record_time(&qc_critical_block, &elapsed);
        if (PARAMETER.circuit.timer) {
            this->totalTimeMs_ += elapsed;
            // printf("\033[1;36mTotal execution time: %.6f ms\033[0m\n", this->totalTimeMs_);
            cout << "\033[1;36mTotal execution time: " << this->totalTimeMs_ << " ms\033[0m" << endl;
        }
        // while (this->simulateStep()){ /* 協調融合しない場合は最後まで */ }
        // cout << "Final state: " << this->finalState_ << endl;
        if (PARAMETER.circuit.verbose) {
            cout << this->layer_.size() << " layers executed." << endl;
            cout << this->wires.size() << " qubits used." << endl;
        }
    }
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
    edges0.insert(edges0.end(), this->numQubits_ - qubitIndex - 1, identityEdge);
    edges1.insert(edges1.end(), this->numQubits_ - qubitIndex - 1, identityEdge);
    QMDDGate m0 = accumulate(edges0.rbegin() + 1, edges0.rend(), edges0.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    });
    QMDDGate m1 = accumulate(edges1.rbegin() + 1, edges1.rend(), edges1.back(), [](const QMDDEdge& accumulated, const QMDDEdge& current) {
        return mathUtils::kron(current, accumulated);
    });
    // QMDDEdge result0 = mathUtils::mul(m0.getInitialEdge(), this->finalState_.getInitialEdge());
    // QMDDEdge result1 = mathUtils::mul(m1.getInitialEdge(), this->finalState_.getInitialEdge());

    QMDDEdge result0 = threadPool.submitFiber([&]() { return mathUtils::mul(m0.getInitialEdge(), this->finalState_.getInitialEdge()); }).get();
    QMDDEdge result1 = threadPool.submitFiber([&]() { return mathUtils::mul(m1.getInitialEdge(), this->finalState_.getInitialEdge()); }).get();

    vector<complex<double>> v0 = result0.getAllElementsForKet();
    vector<complex<double>> v1 = result1.getAllElementsForKet();

    double p0 = mathUtils::sumOfSquares(v0);
    double p1 = mathUtils::sumOfSquares(v1);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 1.0);
    double random_value = dist(gen);

    if (random_value < p0) {
        this->finalState_ = QMDDState(QMDDEdge(result0.weight * (1.0 / sqrt(p0)), make_shared<QMDDNode>(*result0.getStartNode())));
        return 0;
    } else {
        this->finalState_ = QMDDState(QMDDEdge(result0.weight * (1.0 / sqrt(p1)), make_shared<QMDDNode>(*result1.getStartNode())));
        return 1;
    }
}