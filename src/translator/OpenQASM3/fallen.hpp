#ifndef FALLEN_HPP
#define FALLEN_HPP

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <functional>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>

#include "OpenQASM3BaseVisitor.h"
#include "OpenQASM3Parser.h"

using namespace std;

class CircuitGenerator : public OpenQASM3BaseVisitor {
private:
    stringstream circuit_code_;
    map<string, string> gate_mappings_;
    int max_qubit_index_ = -1;

    string quantum_var_ = "q";
    string classical_var_ = "c";
    bool prologue_printed_ = false;

    unordered_map<string, int> classical_sizes_;
    unordered_set<string> emitted_classicals_;

    struct Operation {
        string gate;
        vector<double> params;
        vector<int>    qubits;
    };
    vector<Operation> ops_;

public:
    CircuitGenerator();
    ~CircuitGenerator() override = default;

    string getCircuitCode() const;
    int getMaxQubitIndex() const { return max_qubit_index_; }

    antlrcpp::Any visitGateStmt(OpenQASM3Parser::GateStmtContext *ctx) override;
    antlrcpp::Any visitMeasureStmt(OpenQASM3Parser::MeasureStmtContext *ctx) override;
    antlrcpp::Any visitBarrierStmt(OpenQASM3Parser::BarrierStmtContext *ctx) override;
    antlrcpp::Any visitQregDecl(OpenQASM3Parser::QregDeclContext *ctx) override;
    antlrcpp::Any visitCregDecl(OpenQASM3Parser::CregDeclContext *ctx) override;

    template <typename CircuitT>
    void applyToCircuit(CircuitT& circuit) {
        cout << "[TRANSLATOR] Applying " << ops_.size() << " operations..." << endl;
        for (const auto& op : ops_) {
            const auto& g = op.gate;
            const auto& p = op.params;
            const auto& q = op.qubits;
            try {
                if (g == "H")        circuit.addH(q.at(0));
                else if (g == "X")   circuit.addX(q.at(0));
                else if (g == "Y")   circuit.addY(q.at(0));
                else if (g == "Z")   circuit.addZ(q.at(0));
                else if (g == "S")   circuit.addS(q.at(0));
                else if (g == "T")   circuit.addT(q.at(0));
                else if (g == "SDG") circuit.addSdg(q.at(0));
                else if (g == "TDG") circuit.addTdg(q.at(0));
                else if (g == "SX")  circuit.addV(q.at(0));
                else if (g == "RX")  circuit.addRx(q.at(0), p.at(0));
                else if (g == "RY")  circuit.addRy(q.at(0), p.at(0));
                else if (g == "RZ")  circuit.addRz(q.at(0), p.at(0));
                else if (g == "P")   circuit.addP(q.at(0), p.at(0));
                else if (g == "U1")  circuit.addU1(q.at(0), p.at(0));
                else if (g == "U2")  circuit.addU2(q.at(0), p.at(0), p.at(1));
                else if (g == "U3")  circuit.addU3(q.at(0), p.at(0), p.at(1), p.at(2));
                else if (g == "CX")  circuit.addCX(q.at(0), q.at(1));
                else if (g == "CY")  circuit.addCY(q.at(0), q.at(1));
                else if (g == "CZ")  circuit.addCZ(q.at(0), q.at(1));
                else if (g == "CP")  circuit.addCP(q.at(0), q.at(1), p.at(0));
                else if (g == "CH")  circuit.addCH(q.at(0), q.at(1));
                else if (g == "CRX") circuit.addCRx(q.at(0), q.at(1), p.at(0));
                else if (g == "CRY") circuit.addCRy(q.at(0), q.at(1), p.at(0));
                else if (g == "CRZ") circuit.addCRz(q.at(0), q.at(1), p.at(0));
                else if (g == "CU")  circuit.addCU(q.at(0), q.at(1), p.at(0), p.at(1), p.at(2), p.at(3));
                else if (g == "SWAP") circuit.addSWAP(q.at(0), q.at(1));
                else if (g == "GPHASE") circuit.globalPhase(p.at(0));
                else if (g == "CCX" || g == "CSWAP") {
                    cout << "[EXECUTOR] Skip applying " << g << " (TODO)" << endl;
                } else {
                    cout << "[EXECUTOR] Unknown gate: " << g << " (skipped)" << endl;
                }
            } catch (const exception& ex) {
                cerr << "[EXECUTOR] Error applying " << g << ": " << ex.what() << endl;
            }
        }
        cout << "[TRANSLATOR] Done." << endl;
    }

private:
    void logTranslation(int line, const string& src, const string& dst);
    string extractQubitVar(OpenQASM3Parser::QubitContext *qubit);
    string extractQubitIndex(OpenQASM3Parser::QubitContext *qubit);
    string getGateNameFromContext(OpenQASM3Parser::GateNameContext *gateNameCtx);
    string evaluateExpression(OpenQASM3Parser::ExprContext *expr);
    double evaluateExpressionValue(OpenQASM3Parser::ExprContext *expr, bool& ok);
    vector<string> extractQubitIndices(const vector<OpenQASM3Parser::QubitContext*>& qubits);
    void updateMaxQubitIndex(int index) { if (index > max_qubit_index_) max_qubit_index_ = index; }
    void addGateOperation(const string& gate, const vector<double>& params, const vector<int>& qubits);
    void emitClassicalDeclaration(const string& name, int size, int line, const string& src);
};

#endif