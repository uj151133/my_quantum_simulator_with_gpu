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

#include "OpenQASM3BaseVisitor.h"
#include "OpenQASM3Parser.h"

class CircuitGenerator : public OpenQASM3BaseVisitor {
private:
    std::stringstream circuit_code_;
    std::map<std::string, std::string> gate_mappings_;
    int max_qubit_index_ = -1;

    struct Operation {
        std::string gate;
        std::vector<double> params; // 数値化済みのパラメータ
        std::vector<int>    qubits;
    };
    std::vector<Operation> ops_;

public:
    CircuitGenerator();
    ~CircuitGenerator() override = default;

    std::string getCircuitCode() const { return circuit_code_.str(); }
    int getMaxQubitIndex() const { return max_qubit_index_; }

    // Visitor 実装
    antlrcpp::Any visitGateStmt(OpenQASM3Parser::GateStmtContext *ctx) override;
    antlrcpp::Any visitMeasureStmt(OpenQASM3Parser::MeasureStmtContext *ctx) override;
    antlrcpp::Any visitBarrierStmt(OpenQASM3Parser::BarrierStmtContext *ctx) override;

    // 生成した操作を回路へ適用（テンプレート：Circuit 型に依存）
    template <typename CircuitT>
    void applyToCircuit(CircuitT& circuit) {
        std::cout << "[TRANSLATOR] Applying " << ops_.size() << " operations..." << std::endl;
        for (const auto& op : ops_) {
            const auto& g = op.gate;
            const auto& p = op.params;
            const auto& q = op.qubits;

            try {
                if (g == "H")        { circuit.addH(q.at(0)); }
                else if (g == "X")   { circuit.addX(q.at(0)); }
                else if (g == "Y")   { circuit.addY(q.at(0)); }
                else if (g == "Z")   { circuit.addZ(q.at(0)); }
                else if (g == "S")   { circuit.addS(q.at(0)); }
                else if (g == "T")   { circuit.addT(q.at(0)); }
                else if (g == "SDG") { circuit.addSdg(q.at(0)); }
                else if (g == "TDG") { circuit.addTdg(q.at(0)); }
                else if (g == "SX")  { circuit.addV(q.at(0)); }

                // 1パラメータ1量子ビット
                else if (g == "RX")  { circuit.addRx(p.at(0), q.at(0)); }
                else if (g == "RY")  { circuit.addRy(p.at(0), q.at(0)); }
                else if (g == "RZ")  { circuit.addRz(p.at(0), q.at(0)); }
                else if (g == "P")   { circuit.addP(p.at(0), q.at(0)); }
                else if (g == "U1")  { circuit.addU1(p.at(0), q.at(0)); }

                // 多パラメータ1量子ビット
                else if (g == "U2")  { circuit.addU2(p.at(0), p.at(1), q.at(0)); }
                else if (g == "U3")  { circuit.addU3(p.at(0), p.at(1), p.at(2), q.at(0)); }

                // 2量子ビット（制御）
                else if (g == "CX")  { circuit.addCX(q.at(0), q.at(1)); }
                else if (g == "CY")  { circuit.addCY(q.at(0), q.at(1)); }
                else if (g == "CZ")  { circuit.addCZ(q.at(0), q.at(1)); }
                else if (g == "CP")  { circuit.addCP(p.at(0), q.at(0), q.at(1)); }
                else if (g == "CH")  { circuit.addCH(q.at(0), q.at(1)); }
                else if (g == "CRX") { circuit.addCRx(p.at(0), q.at(0), q.at(1)); }
                else if (g == "CRY") { circuit.addCRy(p.at(0), q.at(0), q.at(1)); }
                else if (g == "CRZ") { circuit.addCRz(p.at(0), q.at(0), q.at(1)); }
                else if (g == "CU")  { circuit.addCU(q.at(0), q.at(1), p.at(0), p.at(1), p.at(2)); }
                else if (g == "SWAP"){ circuit.addSWAP(q.at(0), q.at(1)); }

                // 未対応（シグネチャ不明のため適用スキップ）
                else if (g == "CCX" || g == "CSWAP") {
                    std::cout << "[EXECUTOR] Skip applying " << g << " (TODO: implement mapping)" << std::endl;
                } else if (g == "ID" || g == "GPHASE") {
                    // no-op
                } else {
                    std::cout << "[EXECUTOR] Unknown gate: " << g << " (skipped)" << std::endl;
                }
            } catch (const std::exception& ex) {
                std::cerr << "[EXECUTOR] Error applying " << g << ": " << ex.what() << std::endl;
            }
        }
        std::cout << "[TRANSLATOR] Done." << std::endl;
    }

private:
    std::string getGateNameFromContext(OpenQASM3Parser::GateNameContext *gateNameCtx);
    std::string extractQubitIndex(OpenQASM3Parser::QubitContext *qubit);
    std::string evaluateExpression(OpenQASM3Parser::ExprContext *expr); // 表示用
    double      evaluateExpressionValue(OpenQASM3Parser::ExprContext *expr, bool& ok); // 数値評価
    std::vector<std::string> extractQubitIndices(const std::vector<OpenQASM3Parser::QubitContext*>& qubits);
    void updateMaxQubitIndex(int index) { if (index > max_qubit_index_) max_qubit_index_ = index; }

    void addGateOperation(const std::string& gate,
                          const std::vector<double>& params,
                          const std::vector<int>& qubits);
};

#endif