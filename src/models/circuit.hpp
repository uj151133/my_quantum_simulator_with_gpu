#ifndef CIRCUIT_HPP
#define CIRCUIT_HPP

#include <numeric>
#include <cmath>
#include <queue>
#include <deque>
#include <array>
#include <random>
#include <iostream>
#include <bitset>
#include <ranges>
#include <algorithm>
#include <cctype>
#include "qmdd.hpp"
#include "gate.hpp"
#include "state.hpp"
#include "../common/Core.hpp"
#include "../common/mathUtils.hpp"
#include "../common/bigBang.hpp"
#include "../common/parameter.hpp"
#include "../opt/law.hpp"
#include "../modules/importer.hpp"
#include "../models/dag.hpp"
#include "../models/heauristic.hpp"

using namespace std;

struct Part {
    Type type;
    QMDDGate gate;
};

template<typename T>
vector<T> sorted(const vector<T>& vec) {
    vector<T> result = vec;
    sort(result.begin(), result.end());
    return result;
}

template<typename T, size_t N>
array<T, N> sorted(const array<T, N>& arr) {
    array<T, N> result = arr;
    sort(result.begin(), result.end());
    return result;
}
extern const unordered_set<string> cancer;

class QuantumCircuit {
private:
    vector<vector<Part>> wires;
    vector<vector<QMDDEdge>> layer_;
    vector<Core> irLog_;
    vector<int> phy2log_;
    vector<int> log2phy_;
    vector<int> swapTable_;
    double totalTimeMs_ = .0;
    queue<QMDDGate> gateQueue_;
    QMDDState finalState_;
    int numQubits_;
    bool irEnabled_ = true;
    vector<QMDDGate> pending_;
    vector<Core> metaQueue_;
    size_t execIdx_ = 0;
    unordered_map<uint64_t, QMDDGate> fusedStore_;
    uint64_t nextFusedId_ = 1;

    inline int resolveQubit(int q) const {
        return (this->irEnabled_ ? this->swapTable_[q] : q);
    }
    vector<int> countCancer() const;
    void consult();
    int getMaxDepth(optional<int> start, optional<int> end) const;
    void normalizeLayer();
    void smartInsert(const vector<int>& qubitIndices, const Part& part);
    int searchJOKER(const vector<int>& qubitIndices);
    void build();
    static string upper(string s);
    static bool isDiagTag(const string& tagU);
    void moveQueueToPending();
    void buildMetaFromIR(const vector<Core>& ops);

    void compute();
    void uncompute();

public:

    vector<vector<int>> quantumRegister_;

    void enableIR(bool on);
    void clearIR();
    void emitIR(const vector<Core>& ops);
    // void compileIRWithLaw(const law::Options& opt);
    void scheduleIRWithModel(const string& modelPath = ::SCHEDULER_MODEL_PATH);
    void preprocess(const law::Options& opt, const string& modelPath = ::SCHEDULER_MODEL_PATH);

    vector<Core> snapshotQueueWindow(size_t max_items) const;
    void fuseRanges(const vector<pair<int,int>>& ranges);

    QuantumCircuit(int numQubitits, QMDDState initialState);
    QuantumCircuit(int numQubitits);
    
    ~QuantumCircuit() = default;
    QuantumCircuit(const QuantumCircuit& other) = default;
    QuantumCircuit& operator=(const QuantumCircuit& other) = default;
    QuantumCircuit(QuantumCircuit&& other) = default;
    QuantumCircuit& operator=(QuantumCircuit&& other) = default;

    queue<QMDDGate> getGateQueue() const;
    QMDDState getFinalState() const;
    double getTotalTimeMs() const;

    void setRegister(int registerIdx, int size);
    // void setIrLog(const vector<Core>& ops);

    void addI(int qubitIndex);
    void addPh(int qubitIndex, double delta);
    void addPh(vector<pair<int, double>>& qubitParams);
    void addX(int qubitIndex);
    void addX(vector<int>& qubitIndices);
    void addAllX();
    void addY(int qubitIndex);
    void addY(vector<int>& qubitIndices);
    void addZ(int qubitIndex);
    void addZ(vector<int>& qubitIndices);
    void addS(int qubitIndex);
    void addS(vector<int>& qubitIndices);
    void addSdg(int qubitIndex);
    void addSdg(vector<int>& qubitIndices);
    void addV(int qubitIndex);
    void addV(vector<int>& qubitIndices);
    void addH(int qubitIndex);
    void addH(vector<int>& qubitIndices);
    void addAllH();
    void addCX(int controlIndex, int targetIndex);
    void addVarCX(int controlIndex, int targetIndex);
    void addCY(int controlIndex, int targetIndex);
    void addCZ(int controlIndex, int targetIndex);
    void addDCNOT(int controlIndex, int targetIndex);
    void addSWAP(int qubitIndex1, int qubitIndex2);
    void addiSWAP(int qubitIndex1, int qubitIndex2);
    void addP(int qubitIndex, double phi);
    void addP(vector<pair<int, double>>& qubitParams);
    void addT(int qubitIndex);
    void addT(vector<int>& qubitIndices);
    void addTdg(int qubitIndex);
    void addTdg(vector<int>& qubitIndices);
    void addCP(int controlIndex, int targetIndex, double phi);
    void addCS(int controlIndex, int targetIndex);
    void addR(int qubitIndex, double theta, double phi);
    void addR(vector<pair<int, pair<double, double>>>& qubitParams);
    void addCH(int controlIndex, int targetIndex);
    void addRx(int qubitIndex, double theta);
    // void addRx(vector<pair<int, double>>& qubitParams);
    void addRy(int qubitIndex, double theta);
    // void addRy(vector<pair<int, double>>& qubitParams);
    void addRz(int qubitIndex, double theta);
    void addRz(vector<pair<int, double>>& qubitParams);
    void addRxx(int controlIndex, int targetIndex, double phi);
    void addRyy(int controlIndex, int targetIndex, double phi);
    void addRzz(int controlIndex, int targetIndex, double phi);
    void addRxy(int controlIndex, int targetIndex, double phi);
    void addSquareSWAP(int qubitIndex1, int qubitIndex2);
    void addSquareiSWAP(int qubitIndex1, int qubitIndex2);
    void addSWAPalpha(int qubitIndex1, int qubitIndex2, double alpha);
    void addFREDKIN(int controlIndex, int targetIndex1, int targetIndex2);
    void addU(int qubitIndex, double theta, double phi, double lambda);
    void addU(vector<pair<int, tuple<double, double, double>>>& qubitParams);
    void addU1(int qubitIndex, double lambda);
    void addU1(vector<pair<int, double>>& qubitParams);
    void addU2(int qubitIndex, double phi, double lambda);
    void addU2(vector<pair<int, pair<double, double>>>& qubitParams);
    void addU3(int qubitIndex, double theta, double phi, double lambda);
    void addU3(vector<pair<int, tuple<double, double, double>>>& qubitParams);

    void addCRx(int controlIndex, int targetIndex, double theta);
    void addCRy(int controlIndex, int targetIndex, double theta);
    void addCRz(int controlIndex, int targetIndex, double theta);
    void addCU(int controlIndex, int targetIndex, double theta, double phi, double lambda);
    void addBARENCO(int qubitIndex, double alpha, double phi, double theta);
    void addB(int qubitIndex);
    void addCSX(int controlIndex, int targetIndex);
    void addN(int qubitIndex, double a, double b, double c);
    void addDB(int qubitIndex);
    void addECR(int controlIndex, int targetIndex);
    void addG(int qubitIndex);
    void addM(int qubitIndex);
    void addsyc(int qubitIndex);
    void addCZS(int controlIndex, int targetIndex);
    void addD(int qubitIndex);
    void addRCCX(int controlIndex1, int controlIndex2, int targetIndex);
    void addPG(int controlIndex1, int controlIndex2, int targetIndex);
    void addToff(const array<int, 2>& controlIndexes, int targetIndex);
    void addMCT(const vector<int>& controlIndexes, int targetIndex);
    void addfFredkin(int controlIndex1, int controlIndex2, int targetIndex);

    void addGate(int qubitIndex, const QMDDGate& gate);

    void addQFT(int numQubits);
    void addQFT();

    void addModularExponentiation(int base, int exponent, int modulus);
    void addCModularExponentiation(int controlIndex, int targetIndex, int base, int exponent, int modulus);

    void addOracle(int omega);
    void addDiffuser();

    void addBarrier();
    void reset(int qubitIndex);
    void globalPhase(double lamda);
    int measure(int qubitIndex);
    bool simulateStep();
    void simulate();

    void criticalExecute();
};

#endif