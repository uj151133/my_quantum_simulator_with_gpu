#include "law.hpp"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <cmath>

namespace law {
static inline string upper(string s){for(auto& c:s) c=(char)toupper((unsigned char)c); return s;}
static inline bool isGate(const Op& o, const char* n){ return upper(o.gate_type)==n; }
static inline bool samePair(const Op& a,const Op& b){return a.qubits.size()==2 && b.qubits.size()==2 && a.qubits==b.qubits; }
static inline bool isSingleOn(const Op& o,int q){return o.qubits.size()==1 && o.qubits[0]==q;}
static inline bool nearZero(double x){return abs(x) < 1e-12; }
static inline bool envOn(const char* key, bool deflt){
    if(const char* v=getenv(key)){ string s(v);
        if(s=="0"||s=="false"||s=="FALSE") return false;
        if(s=="1"||s=="true" ||s=="TRUE")  return true;
    }
    return deflt;
}

Options optionsFromEnv(const Options& d){
    Options o=d;
    o.rule_R1_single_axis_fuse = envOn("QMDD_RULE_R1", d.rule_R1_single_axis_fuse);
    o.rule_R2_pair_cancel      = envOn("QMDD_RULE_R2", d.rule_R2_pair_cancel);
    o.rule_R3_phase_gadget     = envOn("QMDD_RULE_R3", d.rule_R3_phase_gadget);
    o.rule_R4_commute_rz_ctrl  = envOn("QMDD_RULE_R4", d.rule_R4_commute_rz_ctrl);
    o.rule_R5_desugar_swap     = envOn("QMDD_RULE_R5", d.rule_R5_desugar_swap);
    o.rule_R6_cx_to_cz_via_h   = envOn("QMDD_PREFER_DIAG", d.rule_R6_cx_to_cz_via_h);
    o.rule_R7_merge_diag_angle = envOn("QMDD_RULE_R7", d.rule_R7_merge_diag_angle);
    o.rule_R8_hcxh_to_cz       = envOn("QMDD_RULE_R8", d.rule_R8_hcxh_to_cz);
    o.rule_R9_conjugation_ids  = envOn("QMDD_RULE_R9", d.rule_R9_conjugation_ids);
    o.rule_R10_bubble_diagonals= envOn("QMDD_RULE_R10", d.rule_R10_bubble_diagonals);
    if(const char* e=getenv("QMDD_REWRITE_ITERS")){ try{ o.iters=max(1, stoi(e)); }catch(...){} }
    return o;
}

// R5
static void desugarSwap(vector<Op>& ops){vector<Op> out; out.reserve(ops.size());
    for(const auto& o: ops){
        if(is_gate(o,"SWAP") && o.qubits.size()==2){
            int a=o.qubits[0], b=o.qubits[1];
            out.push_back(Op{"CX",{a,b}});
            out.push_back(Op{"CX",{b,a}});
            out.push_back(Op{"CX",{a,b}});
        }else out.push_back(o);
    }
    ops.swap(out);
}

// R6
static void CXtoCZviaH(vector<Op>& ops){vector<Op> out; out.reserve(ops.size());
    for(const auto& o: ops){
        if((isGate(o,"CX")||isGate(o,"CNOT")) && o.qubits.size()==2){
            int c=o.qubits[0], t=o.qubits[1];
            out.push_back(Op{"H",{t}});
            out.push_back(Op{"CZ",{c,t}});
            out.push_back(Op{"H",{t}});
        }else out.push_back(o);
    }
    ops.swap(out);
}

// R1(角度合成) + 1Q相殺 + R9(共役恒等)
static void cancelOneQubit(vector<Op>& ops, bool doAngle){
    vector<Op> out; out.reserve(ops.size()); size_t i=0;
    while(i<ops.size()){
        Op a=ops[i]; string an=upper(a.gate_type);
        // angle fuse
        if(doAngle && (an=="RX"||an=="RY"||an=="RZ") && i+1<ops.size()){
            size_t j=i+1; double acc=a.theta;
            while(j<ops.size()){
                const Op& b=ops[j];
                if(upper(b.gate_type)==an && isSingleOn(b, a.qubits[0])){ acc+=b.theta; j++; } else break;
            }
            if(!nearZero(acc)){ a.theta=acc; out.push_back(a); }
            i=j; continue;
        }
        // 1Q相殺
        if((an=="X"||an=="Z"||an=="H") && i+1<ops.size()){
            const Op& b=ops[i+1];
            if(upper(b.gate_type)==an && isSingleOn(b, a.qubits[0])){ i+=2; continue; }
        }
        // R9: H Z H = X, H X H = Z
        if(an=="H" && i+2<ops.size()){
            const Op& b=ops[i+1]; const Op& c=ops[i+2];
            if(isSingleOn(a,a.qubits[0]) && isSingleOn(c,a.qubits[0]) && upper(c.gate_type)=="H"){
                if(isGate(b,"Z") && isSingleOn(b,a.qubits[0])){ out.push_back(Op{"X",{a.qubits[0]}}); i+=3; continue; }
                if(isGate(b,"X") && isSingleOn(b,a.qubits[0])){ out.push_back(Op{"Z",{a.qubits[0]}}); i+=3; continue; }
            }
        }
        // R9: X RZ X = RZ(-θ), Z RX Z = RX(-θ)
        if((an=="X"||an=="Z") && i+2<ops.size()){
            const Op& b=ops[i+1]; const Op& c=ops[i+2];
            if(isSingleOn(a,a.qubits[0]) && isSingleOn(c,a.qubits[0]) && upper(c.gate_type)==an){
                if(an=="X" && isGate(b,"RZ") && isSingleOn(b,a.qubits[0])){ Op rz=b; rz.theta=-rz.theta; out.push_back(rz); i+=3; continue; }
                if(an=="Z" && isGate(b,"RX") && isSingleOn(b,a.qubits[0])){ Op rx=b; rx.theta=-rx.theta; out.push_back(rx); i+=3; continue; }
            }
        }
        out.push_back(a); i++;
    }
    ops.swap(out);
}

// R2(2Q)
static void cancelTwoQubits(vector<Op>& ops){vector<Op> out; out.reserve(ops.size()); size_t i=0;
    while(i<ops.size()){
        const Op& a=ops[i];
        if(i+1<ops.size()){
            const Op& b=ops[i+1];
            if(((isGate(a,"CX")||isGate(a,"CNOT")) && (isGate(b,"CX")||isGate(b,"CNOT")) && samePair(a,b)) ||
                (isGate(a,"CZ") && isGate(b,"CZ") && samePair(a,b))){ i+=2; continue; }
        }
        out.push_back(a); i++;
    }
    ops.swap(out);
}

// R3
static void sandwich_crz(vector<Op>& ops){vector<Op> out; out.reserve(ops.size()); size_t i=0;
    while(i<ops.size()){
        if(i+2<ops.size()){
            const Op& a=ops[i]; const Op& b=ops[i+1]; const Op& c=ops[i+2];
            bool a_cx=(isGate(a,"CX")||isGate(a,"CNOT"));
            bool c_cx=(isGate(c,"CX")||isGate(c,"CNOT"));
            if(a_cx && c_cx && samePair(a,c) && isGate(b,"RZ") && isSingleOn(b, a.qubits[1])){
                out.push_back(Op{"CRZ",{a.qubits[0], a.qubits[1]}, b.theta});
                i+=3; continue;
            }
        }
        out.push_back(ops[i]); i++;
    }
    ops.swap(out);
}

// R4
static void commute_rz_control_with_cx(std::vector<Op>& ops){
    for(size_t i=0;i+1<ops.size();++i){
        Op& a=ops[i]; Op& b=ops[i+1];
        if(isGate(a,"RZ") && (isGate(b,"CX")||isGate(b,"CNOT")) && isSingleOn(a, b.qubits[0])) swap(ops[i], ops[i+1]);
    }
}

// R7
static void merge_diagonal_angles(vector<Op>& ops){vector<Op> out; out.reserve(ops.size()); size_t i=0;
    while(i<ops.size()){
        Op a=ops[i]; std::string an=upper(a.gate_type);
        if((an=="CRZ"||an=="CP") && i+1<ops.size()){
            size_t j=i+1; double acc=a.theta;
            while(j<ops.size()){
                const Op& b=ops[j];
                if(upper(b.gate_type)==an && samePair(a,b)){ acc+=b.theta; j++; } else break;
            }
            if(!nearZero(acc)){ a.theta=acc; out.push_back(a); }
            i=j; continue;
        }
        out.push_back(a); i++;
    }
    ops.swap(out);
}

// R10（安全な可換則：対角×対角、または不交差でのみ交換）
static inline bool is_diag_1q(const Op& o){ auto n=upper(o.gate_type); return (n=="RZ"||n=="U1"||n=="P"||n=="S"||n=="T"||n=="Z"); }
static inline bool is_diag_2q(const Op& o){ auto n=upper(o.gate_type); return (n=="CZ"||n=="CP"||n=="CRZ"||n=="RZZ"); }
static inline bool is_diag(const Op& o){ return is_diag_1q(o) || is_diag_2q(o); }
static inline bool disjoint(const Op& a,const Op& b){
    for(int qa: a.qubits) for(int qb: b.qubits) if(qa==qb) return false;
    return true;
}

static void bubble_diagonals(vector<Op>& ops){
    if(ops.size()<2) return;
    bool changed=true; int passes=0;
    while(changed && passes<2){
        changed=false; passes++;
        for(size_t i=0;i+1<ops.size();++i){
            const Op& a=ops[i]; const Op& b=ops[i+1];
            if(!is_diag(b)) continue;
            if(is_diag(a) || disjoint(a,b)){swap(ops[i], ops[i+1]); changed=true; }
        }
    }
}

vector<Op> optimize(const vector<Op>& in, const Options& o){
    vector<Op> ops=in;
    if(o.rule_R5_desugar_swap)    desugarSwap(ops);
    if(o.rule_R6_cx_to_cz_via_h)  CXtoCZviaH(ops);
    for(int k=0;k<o.iters;k++){
        cancelOneQubit(ops, o.rule_R1_single_axis_fuse);
        if(o.rule_R2_pair_cancel)      cancelTwoQubits(ops);
        if(o.rule_R4_commute_rz_ctrl)  commute_rz_control_with_cx(ops);
        if(o.rule_R3_phase_gadget)     sandwich_crz(ops);
        if(o.rule_R7_merge_diag_angle) merge_diagonal_angles(ops);
        if(o.rule_R8_hcxh_to_cz)       /*already included as separate*/ (void)0;
        if(o.rule_R10_bubble_diagonals)bubble_diagonals(ops);
        cancelOneQubit(ops, o.rule_R1_single_axis_fuse);
        if(o.rule_R2_pair_cancel)      cancelTwoQubits(ops);
    }
    return ops;
}

}