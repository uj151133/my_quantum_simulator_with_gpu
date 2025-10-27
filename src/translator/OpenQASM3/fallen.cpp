#include "fallen.hpp"
#include <cmath>

// ゲート名マッピングはここでセットアップ（生成コード用）
CircuitGenerator::CircuitGenerator() {
    // 単一量子ビット
    gate_mappings_["X"] = "addX";
    gate_mappings_["Y"] = "addY";
    gate_mappings_["Z"] = "addZ";
    gate_mappings_["H"] = "addH";
    gate_mappings_["S"] = "addS";
    gate_mappings_["T"] = "addT";
    gate_mappings_["SDG"] = "addSdg";
    gate_mappings_["TDG"] = "addTdg";
    gate_mappings_["SX"] = "addSX";

    // パラメータ付き
    gate_mappings_["RX"] = "addRx";
    gate_mappings_["RY"] = "addRy";
    gate_mappings_["RZ"] = "addRz";
    gate_mappings_["P"]  = "addP";
    gate_mappings_["U1"] = "addU1";
    gate_mappings_["U2"] = "addU2";
    gate_mappings_["U3"] = "addU3";

    // 制御
    gate_mappings_["CX"] = "addCX";
    gate_mappings_["CY"] = "addCY";
    gate_mappings_["CZ"] = "addCZ";
    gate_mappings_["CP"] = "addCP";
    gate_mappings_["CH"] = "addCH";
    gate_mappings_["CRX"] = "addCRx";
    gate_mappings_["CRY"] = "addCRy";
    gate_mappings_["CRZ"] = "addCRz";
    gate_mappings_["CU"]  = "addCU";

    // その他
    gate_mappings_["SWAP"]  = "addSWAP";
    gate_mappings_["CCX"]   = "addToff";
    gate_mappings_["CSWAP"] = "addCSWAP";
    gate_mappings_["ID"]     = "addI";
    gate_mappings_["GPHASE"] = "addGlobalPhase";
}

std::string CircuitGenerator::getGateNameFromContext(OpenQASM3Parser::GateNameContext *g) {
    if (g->ID())     return "ID";
    if (g->X())      return "X";
    if (g->Y())      return "Y";
    if (g->Z())      return "Z";
    if (g->H())      return "H";
    if (g->S())      return "S";
    if (g->SDG())    return "SDG";
    if (g->T())      return "T";
    if (g->TDG())    return "TDG";
    if (g->SX())     return "SX";
    if (g->RX())     return "RX";
    if (g->RY())     return "RY";
    if (g->RZ())     return "RZ";
    if (g->P())      return "P";
    if (g->U1())     return "U1";
    if (g->U2())     return "U2";
    if (g->U3())     return "U3";
    if (g->CX())     return "CX";
    if (g->CY())     return "CY";
    if (g->CZ())     return "CZ";
    if (g->CP())     return "CP";
    if (g->CH())     return "CH";
    if (g->CRX())    return "CRX";
    if (g->CRY())    return "CRY";
    if (g->CRZ())    return "CRZ";
    if (g->CU())     return "CU";
    if (g->SWAP())   return "SWAP";
    if (g->CCX())    return "CCX";
    if (g->CSWAP())  return "CSWAP";
    if (g->GPHASE()) return "GPHASE";
    return "UNKNOWN";
}

antlrcpp::Any CircuitGenerator::visitGateStmt(OpenQASM3Parser::GateStmtContext *ctx) {
    const std::string gate = getGateNameFromContext(ctx->gateName());
    std::cout << "[TRANSLATOR] gate=" << gate;

    // パラメータ（文字列と数値）
    std::vector<std::string> params_text;
    std::vector<double>      params_value;
    if (ctx->paramList()) {
        std::cout << " params=[";
        const auto& es = ctx->paramList()->expr();
        for (size_t i = 0; i < es.size(); ++i) {
            auto* e = es[i];
            bool ok = false;
            const std::string t = evaluateExpression(e);
            const double      v = evaluateExpressionValue(e, ok);
            params_text.push_back(t);
            if (ok) params_value.push_back(v);
            else    params_value.push_back(NAN);
            std::cout << t << (i + 1 < es.size() ? ", " : "");
        }
        std::cout << "]";
    }

    // 量子ビット
    std::vector<std::string> qubits_s = extractQubitIndices(ctx->gateArgs()->qubit());
    std::vector<int> qubits;
    std::cout << " qubits=[";
    for (size_t i = 0; i < qubits_s.size(); ++i) {
        std::cout << qubits_s[i] << (i + 1 < qubits_s.size() ? ", " : "");
        try {
            int qi = std::stoi(qubits_s[i]);
            qubits.push_back(qi);
            updateMaxQubitIndex(qi);
        } catch (...) {
            qubits.push_back(0);
        }
    }
    std::cout << "]" << std::endl;

    // コード生成（文字列）
    auto it = gate_mappings_.find(gate);
    if (it == gate_mappings_.end()) {
        std::cerr << "[TRANSLATOR] Unknown gate mapping: " << gate << std::endl;
        return nullptr;
    }
    std::string code = "circuit." + it->second + "(";
    for (size_t i = 0; i < params_text.size(); ++i) {
        code += params_text[i];
        if (i + 1 < params_text.size() || !qubits_s.empty()) code += ", ";
    }
    for (size_t i = 0; i < qubits_s.size(); ++i) {
        code += qubits_s[i];
        if (i + 1 < qubits_s.size()) code += ", ";
    }
    code += ");";
    std::cout << "[TRANSLATOR] Generated: " << code << std::endl;
    circuit_code_ << code << '\n';

    // 実行用オペレーションを保存（数値パラメータが NaN の場合はログのみで適用スキップ）
    bool has_nan = false;
    for (double v : params_value) if (std::isnan(v)) { has_nan = true; break; }
    if (!has_nan) {
        addGateOperation(gate, params_value, qubits);
    } else if (!params_value.empty()) {
        std::cout << "[TRANSLATOR] Skip applying " << gate << " due to non-numeric parameter(s)" << std::endl;
    } else {
        addGateOperation(gate, params_value, qubits);
    }

    return nullptr;
}

antlrcpp::Any CircuitGenerator::visitMeasureStmt(OpenQASM3Parser::MeasureStmtContext *ctx) {
    std::string q = extractQubitIndex(ctx->qubit());
    std::cout << "[TRANSLATOR] measure qubit=" << q << std::endl;
    try { updateMaxQubitIndex(std::stoi(q)); } catch (...) {}
    std::string code = "circuit.measure(" + q + ");";
    circuit_code_ << code << '\n';
    // 実運用で測定適用が必要なら、ここで ops_ に push してください（回路APIに依存）
    return nullptr;
}

antlrcpp::Any CircuitGenerator::visitBarrierStmt(OpenQASM3Parser::BarrierStmtContext *ctx) {
    std::vector<std::string> qs_s = extractQubitIndices(ctx->qubitList()->qubit());
    std::vector<int> qs;
    std::cout << "[TRANSLATOR] barrier qubits=[";
    for (size_t i = 0; i < qs_s.size(); ++i) {
        std::cout << qs_s[i] << (i + 1 < qs_s.size() ? ", " : "");
        try { int v = std::stoi(qs_s[i]); qs.push_back(v); updateMaxQubitIndex(v); } catch (...) {}
    }
    std::cout << "]" << std::endl;

    std::string code = "circuit.barrier(";
    for (size_t i = 0; i < qs_s.size(); ++i) {
        code += qs_s[i];
        if (i + 1 < qs_s.size()) code += ", ";
    }
    code += ");";
    std::cout << "[TRANSLATOR] Generated: " << code << std::endl;
    circuit_code_ << code << '\n';
    // バリアはシミュレータ側でno-opなら ops_ へは追加しない
    return nullptr;
}

std::string CircuitGenerator::extractQubitIndex(OpenQASM3Parser::QubitContext *q) {
    if (q && q->NUMBER()) return q->NUMBER()->getText();
    return "0";
}

std::string CircuitGenerator::evaluateExpression(OpenQASM3Parser::ExprContext *e) {
    if (e->NUMBER()) return e->NUMBER()->getText();
    if (e->IDSTR())  return e->IDSTR()->getText();
    if (e->getText() == "pi") return "M_PI";
    if (e->expr().size() == 2) {
        std::string l = evaluateExpression(e->expr(0));
        std::string r = evaluateExpression(e->expr(1));
        std::string op = e->children[1]->getText();
        return "(" + l + " " + op + " " + r + ")";
    }
    return e->getText();
}

// 数値評価（pi と + - * / と () のみ対応）
double CircuitGenerator::evaluateExpressionValue(OpenQASM3Parser::ExprContext *e, bool& ok) {
    ok = true;
    if (e->NUMBER()) {
        try { return std::stod(e->NUMBER()->getText()); }
        catch (...) { ok = false; return NAN; }
    }
    if (e->getText() == "pi") return M_PI;
    if (e->expr().size() == 2) {
        bool okL=false, okR=false;
        double l = evaluateExpressionValue(e->expr(0), okL);
        double r = evaluateExpressionValue(e->expr(1), okR);
        if (!(okL && okR)) { ok = false; return NAN; }
        std::string op = e->children[1]->getText();
        if (op == "+") return l + r;
        if (op == "-") return l - r;
        if (op == "*") return l * r;
        if (op == "/") return l / r;
        ok = false; return NAN;
    }
    if (e->expr().size() == 1) {
        // (expr)
        return evaluateExpressionValue(e->expr(0), ok);
    }
    // 変数など未対応
    ok = false;
    return NAN;
}

std::vector<std::string> CircuitGenerator::extractQubitIndices(const std::vector<OpenQASM3Parser::QubitContext*>& qs) {
    std::vector<std::string> out;
    out.reserve(qs.size());
    for (auto* q : qs) out.push_back(extractQubitIndex(q));
    return out;
}

void CircuitGenerator::addGateOperation(const std::string& gate,
                                        const std::vector<double>& params,
                                        const std::vector<int>& qubits) {
    ops_.push_back(Operation{gate, params, qubits});
}