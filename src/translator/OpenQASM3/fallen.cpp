#include "fallen.hpp"
#include <cctype>

namespace {
int parseBracketSize(const std::string& text, int fallback = 1) {
    auto lb = text.find('[');
    auto rb = text.find(']', lb);
    if (lb != std::string::npos && rb != std::string::npos && rb > lb + 1) {
        try {
            return std::stoi(text.substr(lb + 1, rb - lb - 1));
        } catch (...) {}
    }
    return fallback;
}
}

// マッピング初期化（生成コードでも使用）
CircuitGenerator::CircuitGenerator() {
    gate_mappings_["X"] = "addX";
    gate_mappings_["Y"] = "addY";
    gate_mappings_["Z"] = "addZ";
    gate_mappings_["H"] = "addH";
    gate_mappings_["S"] = "addS";
    gate_mappings_["T"] = "addT";
    gate_mappings_["SDG"] = "addSdg";
    gate_mappings_["TDG"] = "addTdg";
    gate_mappings_["SX"] = "addV";

    gate_mappings_["RX"] = "addRx";
    gate_mappings_["RY"] = "addRy";
    gate_mappings_["RZ"] = "addRz";
    gate_mappings_["P"]  = "addP";
    gate_mappings_["U1"] = "addU1";
    gate_mappings_["U2"] = "addU2";
    gate_mappings_["U3"] = "addU3";

    gate_mappings_["CX"] = "addCX";
    gate_mappings_["CY"] = "addCY";
    gate_mappings_["CZ"] = "addCZ";
    gate_mappings_["CP"] = "addCP";
    gate_mappings_["CH"] = "addCH";
    gate_mappings_["CRX"] = "addCRx";
    gate_mappings_["CRY"] = "addCRy";
    gate_mappings_["CRZ"] = "addCRz";
    gate_mappings_["CU"]  = "addCU";

    gate_mappings_["SWAP"]  = "addSWAP";
    gate_mappings_["CCX"]   = "addToff";
    gate_mappings_["CSWAP"] = "addCSWAP";
    gate_mappings_["ID"]    = "addI";
    gate_mappings_["GPHASE"] = "globalPhase";
}

string CircuitGenerator::getCircuitCode() const {
    string code = circuit_code_.str();
    const int inferred = max_qubit_index_ + 1;
    if (inferred > 0) {
        const string needle = "QuantumCircuit " + quantum_var_ + "(";
        const size_t pos = code.find(needle);
        if (pos != string::npos) {
            const size_t sizePos = pos + needle.size();
            const size_t end = code.find(')', sizePos);
            if (end != string::npos) {
                code.replace(sizePos, end - sizePos, to_string(inferred));
            }
        }
    }
    return code;
}

static const char* pickContrastFgColor() {
    static bool inited = false;
    static const char* code = "\x1b[90m";
    if (inited) return code;
    inited = true;

    if (const char* env = getenv("QASM_TRANSLATE_COLOR")) {
        if (!strcmp(env, "white")) return code = "\x1b[97m";
        if (!strcmp(env, "black")) return code = "\x1b[30m";
        if (!strcmp(env, "gray") || !strcmp(env, "grey")) return code = "\x1b[90m";
    }
    if (const char* cfg = getenv("COLORFGBG")) {
        string s(cfg);
        size_t pos = s.rfind(';');
        int bg = -1;
        if (pos != string::npos) {
            try { bg = stoi(s.substr(pos + 1)); } catch (...) {}
        }
        if (bg >= 0) code = (bg <= 7) ? "\x1b[97m" : "\x1b[30m";
    }
    return code;
}

void CircuitGenerator::logTranslation(int line, const string& src, const string& dst) {
    static const char* COL = pickContrastFgColor();
    static const char* RST = "\x1b[0m";
    if (isatty(fileno(stdout))) {
        cout << COL << "[L" << line << "] " << src << "  =>  " << dst << RST << endl;
    } else {
        cout << "[L" << line << "] " << src << "  =>  " << dst << endl;
    }
}

string CircuitGenerator::extractQubitVar(OpenQASM3Parser::QubitContext *q) {
    if (!q) return "q";
    string t = q->getText();
    auto pos = t.find('[');
    if (pos != string::npos && pos > 0) return t.substr(0, pos);
    for (char c : t) if (isalpha(static_cast<unsigned char>(c)) || c == '_') return "q";
    return "q";
}

string CircuitGenerator::extractQubitIndex(OpenQASM3Parser::QubitContext *q) {
    if (!q) return "0";
    string t = q->getText();
    auto l = t.find('[');
    auto r = t.find(']');
    if (l != string::npos && r != string::npos && r > l + 1) return t.substr(l + 1, r - l - 1);
    return "0";
}

string CircuitGenerator::getGateNameFromContext(OpenQASM3Parser::GateNameContext *g) {
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
    const string gate = getGateNameFromContext(ctx->gateName());

    vector<string> params_text;
    vector<double> params_value;
    if (ctx->paramList()) {
        const auto& es = ctx->paramList()->expr();
        for (auto* e : es) {
            bool ok = false;
            params_text.push_back(evaluateExpression(e));
            params_value.push_back(evaluateExpressionValue(e, ok));
            if (!ok) params_value.back() = NAN;
        }
    }

    vector<string> qubits_s;
    vector<int> qubits;
    string local_qvar = quantum_var_;
    if (ctx->gateArgs()) {
        const auto& qb = ctx->gateArgs()->qubit();
        if (!qb.empty()) {
            local_qvar = extractQubitVar(qb[0]);
            quantum_var_ = local_qvar;
        }
        for (auto* qctx : qb) {
            string sidx = extractQubitIndex(qctx);
            qubits_s.push_back(sidx);
            try { int qi = stoi(sidx); qubits.push_back(qi); updateMaxQubitIndex(qi); }
            catch (...) { qubits.push_back(0); }
        }
    }

    auto q0 = qubits_s.size() > 0 ? qubits_s[0] : "0";
    auto q1 = qubits_s.size() > 1 ? qubits_s[1] : "0";
    auto P  = [&](size_t i){ return i < params_text.size() ? params_text[i] : string("0"); };

    string code;
    if (gate == "GPHASE") {
        code = quantum_var_ + ".globalPhase(" + P(0) + ");";
    } else if (gate == "H" || gate == "X" || gate == "Y" || gate == "Z" ||
               gate == "S" || gate == "T" || gate == "SDG" || gate == "TDG" ||
               gate == "SX" || gate == "ID") {
        code = local_qvar + "." + gate_mappings_[gate] + "(" + q0 + ");";
    } else if (gate == "RX" || gate == "RY" || gate == "RZ" || gate == "P" || gate == "U1") {
        code = local_qvar + "." + gate_mappings_[gate] + "(" + q0 + ", " + P(0) + ");";
    } else if (gate == "U2") {
        code = local_qvar + ".addU2(" + q0 + ", " + P(0) + ", " + P(1) + ");";
    } else if (gate == "U3") {
        code = local_qvar + ".addU3(" + q0 + ", " + P(0) + ", " + P(1) + ", " + P(2) + ");";
    } else if (gate == "CX" || gate == "CY" || gate == "CZ" || gate == "CH" || gate == "SWAP") {
        code = local_qvar + "." + gate_mappings_[gate] + "(" + q0 + ", " + q1 + ");";
    } else if (gate == "CP") {
        code = local_qvar + ".addCP(" + q0 + ", " + q1 + ", " + P(0) + ");";
    } else if (gate == "CRX" || gate == "CRY" || gate == "CRZ") {
        code = local_qvar + "." + gate_mappings_[gate] + "(" + q0 + ", " + q1 + ", " + P(0) + ");";
    } else if (gate == "CU") {
        code = local_qvar + ".addCU(" + q0 + ", " + q1 + ", " + P(0) + ", " + P(1) + ", " + P(2) + ");";
    } else {
        return nullptr;
    }

    if (!prologue_printed_) {
        int nq = max(0, max_qubit_index_ + 1);
        logTranslation(ctx->start ? ctx->start->getLine() : 0,
                       "qreg " + quantum_var_ + "[" + to_string(nq) + "];",
                       "QuantumCircuit " + quantum_var_ + "(" + to_string(nq) + ");");
        prologue_printed_ = true;
    }

    string qasm_src = ctx->getText();
    logTranslation(ctx->start ? ctx->start->getLine() : 0, qasm_src, code);
    circuit_code_ << code << '\n';

    bool has_nan = false;
    for (double v : params_value) {
        if (isnan(v)) { has_nan = true; break; }
    }
    if (!has_nan || params_value.empty()) addGateOperation(gate, params_value, qubits);

    return nullptr;
}

antlrcpp::Any CircuitGenerator::visitMeasureStmt(OpenQASM3Parser::MeasureStmtContext *ctx) {
    string qidx = extractQubitIndex(ctx->qubit());
    string qvar = extractQubitVar(ctx->qubit());
    quantum_var_ = qvar;
    try { updateMaxQubitIndex(stoi(qidx)); } catch (...) {}

    string full = ctx->getText();
    string cvar = classical_var_;
    string cidx = "0";
    auto arrow = full.find("->");
    if (arrow != string::npos) {
        size_t start = arrow + 2;
        size_t lb = full.find('[', start);
        size_t rb = full.find(']', lb);
        if (lb != string::npos && rb != string::npos && rb > lb + 1) {
            cvar = full.substr(start, lb - start);
            cidx = full.substr(lb + 1, rb - lb - 1);
        } else {
            size_t end = full.find(';', start);
            cvar = full.substr(start, (end == string::npos ? full.size() : end) - start);
        }
    }
    classical_var_ = cvar;

    int idxVal = 0;
    try { idxVal = stoi(cidx); } catch (...) { idxVal = 0; }
    int required = idxVal + 1;

    auto it = classical_sizes_.find(cvar);
    if (it == classical_sizes_.end()) {
        emitClassicalDeclaration(cvar, required,
                                 ctx->getStart()->getLine(),
                                 "creg " + cvar + "[" + to_string(required) + "];");
    } else if (required > it->second) {
        classical_sizes_[cvar] = required;
    }

    string code = cvar + "[" + cidx + "] = " + qvar + ".measure(" + qidx + ");";
    logTranslation(ctx->getStart()->getLine(), full, code);
    circuit_code_ << code << '\n';
    return nullptr;
}

antlrcpp::Any CircuitGenerator::visitBarrierStmt(OpenQASM3Parser::BarrierStmtContext *ctx) {
    vector<string> qs_s = extractQubitIndices(ctx->qubitList()->qubit());
    for (auto& s : qs_s) {
        try { int v = stoi(s); updateMaxQubitIndex(v); } catch (...) {}
    }

    string code = quantum_var_ + ".barrier(";
    for (size_t i = 0; i < qs_s.size(); ++i) {
        code += qs_s[i];
        if (i + 1 < qs_s.size()) code += ", ";
    }
    code += ");";

    string src = ctx->getText();
    logTranslation(ctx->getStart()->getLine(), src, code);
    circuit_code_ << code << '\n';
    return nullptr;
}

antlrcpp::Any CircuitGenerator::visitQregDecl(OpenQASM3Parser::QregDeclContext* ctx) {
    if (!ctx || !ctx->IDSTR()) return visitChildren(ctx);

    const string var = ctx->IDSTR()->getText();
    const int qubitCount = max(1, parseBracketSize(ctx->getText(), 1));

    quantum_var_ = var;
    max_qubit_index_ = max(max_qubit_index_, qubitCount - 1);
    prologue_printed_ = true;

    const string dst = "QuantumCircuit " + var + "(" + to_string(qubitCount) + ");";
    logTranslation(ctx->getStart()->getLine(), ctx->getText(), dst);
    circuit_code_ << dst << '\n';
    return nullptr;
}

antlrcpp::Any CircuitGenerator::visitCregDecl(OpenQASM3Parser::CregDeclContext* ctx) {
    if (!ctx || !ctx->IDSTR()) return visitChildren(ctx);

    const string var = ctx->IDSTR()->getText();
    const int count = max(1, parseBracketSize(ctx->getText(), 1));

    classical_var_ = var;
    emitClassicalDeclaration(var, count, ctx->getStart()->getLine(), ctx->getText());
    return nullptr;
}

string CircuitGenerator::evaluateExpression(OpenQASM3Parser::ExprContext *e) {
    if (!e) return "0";
    const string text = e->getText();
    if (text == "pi") return "M_PI";
    if (e->expr().size() == 2) {
        string l = evaluateExpression(e->expr(0));
        string r = evaluateExpression(e->expr(1));
        string op = e->children[1]->getText();
        return "(" + l + " " + op + " " + r + ")";
    }
    if (e->expr().size() == 1) {
        return evaluateExpression(e->expr(0));
    }
    return text;
}

double CircuitGenerator::evaluateExpressionValue(OpenQASM3Parser::ExprContext *e, bool& ok) {
    ok = false;
    if (!e) return 0.0;

    const string text = e->getText();
    if (text == "pi") {
        ok = true;
        return M_PI;
    }

    if (e->expr().size() == 2) {
        bool okL = false, okR = false;
        double l = evaluateExpressionValue(e->expr(0), okL);
        double r = evaluateExpressionValue(e->expr(1), okR);
        if (!(okL && okR)) { ok = false; return NAN; }
        string op = e->children[1]->getText();
        ok = true;
        if (op == "+") return l + r;
        if (op == "-") return l - r;
        if (op == "*") return l * r;
        if (op == "/") return l / r;
        ok = false;
        return NAN;
    }

    if (e->expr().size() == 1) {
        double val = evaluateExpressionValue(e->expr(0), ok);
        return val;
    }

    try {
        size_t consumed = 0;
        double val = stod(text, &consumed);
        if (consumed == text.size()) {
            ok = true;
            return val;
        }
    } catch (...) {}
    ok = false;
    return NAN;
}

vector<string> CircuitGenerator::extractQubitIndices(const vector<OpenQASM3Parser::QubitContext*>& qs) {
    vector<string> out;
    out.reserve(qs.size());
    for (auto* q : qs) out.push_back(extractQubitIndex(q));
    return out;
}

void CircuitGenerator::addGateOperation(const string& gate,
                                        const vector<double>& params,
                                        const vector<int>& qubits) {
    ops_.push_back(Operation{gate, params, qubits});
}

void CircuitGenerator::emitClassicalDeclaration(const string& name, int size, int line, const string& src) {
    if (size <= 0) size = 1;
    classical_sizes_[name] = max(classical_sizes_[name], size);
    if (emitted_classicals_.count(name)) return;
    const string dst = "int " + name + "[" + to_string(size) + "] = {0};";
    logTranslation(line, src, dst);
    circuit_code_ << dst << '\n';
    emitted_classicals_.insert(name);
}