#ifndef CIRCUIT_HPP
#define CIRCUIT_HPP

#include <numeric>
#include <queue>
#include <array>
#include <random>
#include <iostream>
#include "qmdd.hpp"
#include "gate.hpp"
#include "state.hpp"
#include "../common/mathUtils.hpp"
#include "../common/constant.hpp"

using namespace std;

class QuantumCircuit {
private:
    queue<QMDDGate> gateQueue;
    QMDDState finalState;
    int numQubits;

public:
    QuantumCircuit(int numQubitits, QMDDState initialState);
    QuantumCircuit(int numQubitits);
    ~QuantumCircuit() = default;
    queue<QMDDGate> getGateQueue() const;
    QMDDState getFinalState() const;
    QuantumCircuit(const QuantumCircuit& other) = default;
    QuantumCircuit& operator=(const QuantumCircuit& other) = default;
    QuantumCircuit(QuantumCircuit&& other) = default;
    QuantumCircuit& operator=(QuantumCircuit&& other) = default;

    void addI(int qubitIndex);
    void addPh(int qubitIndex, double delta);
    void addX(int qubitIndex);
    void addAllX();
    void addY(int qubitIndex);
    void addZ(int qubitIndex);
    void addS(int qubitIndex);
    void addV(int qubitIndex);
    void addH(int qubitIndex);
    void addAllH();
    void addCX(int controlIndex, int targetIndex);
    void addVarCX(int controlIndex, int targetIndex);
    void addCZ(int controlIndex, int targetIndex);
    void addDCNOT(int controlIndex, int targetIndex);
    void addSWAP(int qubitIndex1, int qubitIndex2);
    void addiSWAP(int qubitIndex1, int qubitIndex2);
    void addP(int qubitIndex, double phi);
    void addT(int qubitIndex);
    void addCP(int controlIndex, int targetIndex, double phi);
    void addCS(int controlIndex, int targetIndex);
    void addRx(int qubitIndex, double theta);
    void addRy(int qubitIndex, double theta);
    void addRz(int qubitIndex, double theta);
    void addRxx(int controlIndex, int targetIndex, double phi);
    void addRyy(int controlIndex, int targetIndex, double phi);
    void addRzz(int controlIndex, int targetIndex, double phi);
    void addRxy(int controlIndex, int targetIndex, double phi);
    void addSquareSWAP(int qubitIndex1, int qubitIndex2);
    void addSquareiSWAP(int qubitIndex1, int qubitIndex2);
    void addSWAPalpha(int qubitIndex1, int qubitIndex2, double alpha);
    void addFREDKIN(int controlIndex, int targetIndex1, int targetIndex2);
    void addU(int qubitIndex, double theta, double phi, double lambda);
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
    void addToff(vector<int>& controlIndexes, int targetIndex);
    void addToff2(array<int, 2>& controlIndexes, int targetIndex);
    void addfFredkin(int controlIndex1, int controlIndex2, int targetIndex);

    void addGate(int qubitIndex, const QMDDGate& gate);

    void addOracle(int omega);
    void addIAM();

    void execute();
    QMDDState read(int qubitIndex);
    // コンストラクタやその他のメンバ関数はここに追加できます
};

#endif