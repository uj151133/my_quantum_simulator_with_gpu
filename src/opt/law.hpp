#ifndef LAW_HPP
#define LAW_HPP

#include <string>
#include <vector>

using namespace std;
namespace law {

struct Op {
    string gate_type;
    vector<int> qubits;
    double theta = .0, phi = .0, lamda = .0;
};

struct Options {
    bool rule_R1_single_axis_fuse = false; // 連続RX/RY/RZの角度合成（既定OFF）
    bool rule_R2_pair_cancel      = true;  // X/X, Z/Z, H/H, CX/CX, CZ/CZ 相殺
    bool rule_R3_phase_gadget     = true;  // CX - RZ(t) - CX → CRZ
    bool rule_R4_commute_rz_ctrl  = true;  // RZ(control) と CX の交換
    bool rule_R5_desugar_swap     = true;  // SWAP → CX3
    bool rule_R6_cx_to_cz_via_h   = true; // CX → H-CZ-H（任意でON）
    bool rule_R7_merge_diag_angle = false; // CRZ/CP の角度合成（既定OFF）
    bool rule_R8_hcxh_to_cz       = true;  // H-CX-H → CZ
    bool rule_R9_conjugation_ids  = true;  // HZH=X, HXH=Z, X RZ X=RZ(-θ), Z RX Z=RX(-θ)
    bool rule_R10_bubble_diagonals= true;  // 対角クラスタリング（安全可換則）
    int  iters = 2;                        // 固定点反復回数（QMDD_REWRITE_ITERSで上書き可）
};

vector<Op> optimize(const vector<Op>& in, const Options& opt);
Options optionsFromEnv(const Options& defaults = Options());

}

#endif