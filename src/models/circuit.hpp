#ifndef CIRCUIT_HPP
#define CIRCUIT_HPP

#include <numeric>
#include <queue>
#include <array>
#include <random>
#include <iostream>
#include <bitset>
#include <ranges>
#include "qmdd.hpp"
#include "gate.hpp"
#include "state.hpp"
#include "../common/mathUtils.hpp"
#include "../common/constant.hpp"

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

class QuantumCircuit {
private:
    vector<vector<Part>> wires;
    queue<QMDDGate> layer;
    QMDDState finalState;
    int numQubits;
    int getMaxDepth(optional<int> start, optional<int> end) const;
    void normalizeLayer();

public:
    QuantumCircuit(int numQubitits, QMDDState initialState);
    QuantumCircuit(int numQubitits);
    ~QuantumCircuit() = default;
    queue<QMDDGate> getLayer() const;
    QMDDState getFinalState() const;
    QuantumCircuit(const QuantumCircuit& other) = default;
    QuantumCircuit& operator=(const QuantumCircuit& other) = default;
    QuantumCircuit(QuantumCircuit&& other) = default;
    QuantumCircuit& operator=(QuantumCircuit&& other) = default;

    void addI(int qubitIndex);
    void addPh(const vector<pair<int, double>>& qubitParams);
    void addX(const vector<int>& qubitIndices);
    void addAllX();
    void addY(const vector<int>& qubitIndices);
    void addZ(const vector<int>& qubitIndices);
    void addS(const vector<int>& qubitIndices);
    void addSdg(const vector<int>& qubitIndices);
    void addV(const vector<int>& qubitIndices);
    void addH(const vector<int>& qubitIndices);
    void addAllH();
    void addCX(int controlIndex, int targetIndex);
    void addVarCX(int controlIndex, int targetIndex);
    void addCZ(int controlIndex, int targetIndex);
    void addDCNOT(int controlIndex, int targetIndex);
    void addSWAP(int qubitIndex1, int qubitIndex2);
    void addiSWAP(int qubitIndex1, int qubitIndex2);
    void addP(const vector<pair<int, double>>& qubitParams);
    void addT(const vector<int>& qubitIndices);
    void addTdg(const vector<int>& qubitIndices);
    void addCP(int controlIndex, int targetIndex, double phi);
    void addCS(int controlIndex, int targetIndex);
    void addR(const vector<pair<int, pair<double, double>>>& qubitParams);
    void addRx(const vector<pair<int, double>>& qubitParams);
    void addRy(const vector<pair<int, double>>& qubitParams);
    void addRz(const vector<pair<int, double>>& qubitParams);
    void addRxx(int controlIndex, int targetIndex, double phi);
    void addRyy(int controlIndex, int targetIndex, double phi);
    void addRzz(int controlIndex, int targetIndex, double phi);
    void addRxy(int controlIndex, int targetIndex, double phi);
    void addSquareSWAP(int qubitIndex1, int qubitIndex2);
    void addSquareiSWAP(int qubitIndex1, int qubitIndex2);
    void addSWAPalpha(int qubitIndex1, int qubitIndex2, double alpha);
    void addFREDKIN(int controlIndex, int targetIndex1, int targetIndex2);
    void addU(const vector<pair<int, tuple<double, double, double>>>& qubitParams);
    void addU1(const vector<pair<int, double>>& qubitParams);
    void addU2(const vector<pair<int, pair<double, double>>>& qubitParams);
    void addU3(const vector<pair<int, tuple<double, double, double>>>& qubitParams);
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

    void addOracle(int omega);
    void addIAM();

    void addBarrier();

    void simulate();
    int measure(int qubitIndex);
};

#endif