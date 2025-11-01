#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cctype>

using namespace std;

struct Core {
    // 共通タグ（IRでも実行時でも使用）
    // 例: "H","RZ","RX","CZ","CX","CP","CRZ","FUSED" など
    string tag;

    // 作用量子ビット（1Q/2Q/FUSEDは集合）
    vector<int> qubits;

    // パラメータ（1Q/CR*系など）
    double theta  = .0;
    double phi    = .0;
    double lambda = .0;

    // 特徴（並べ替え/融合の判断で利用）
    bool isDiag  = false;   // 対角か（RZ,U1,P,S,T,Z,CZ,CP,CRZ,RZZなど）
    bool isFused = false;   // FUSED生成か
    uint64_t handle = 0;     // FUSED実体のストアID（非FUSEDは0）
    size_t edgeNodes = 0;   // このゲートのQMDDノード数（任意）

    static string upper(string s) {
        for (auto& c : s) c = (char)toupper((unsigned char)c);
        return s;
    }
    static bool isDiagTag(const string& tagU) {
        const string t = tagU;
        if (t=="RZ"||t=="U1"||t=="P"||t=="S"||t=="T"||t=="Z") return true;
        if (t=="CZ"||t=="CP"||t=="CRZ"||t=="RZZ") return true;
        return false;
    }
    // 正規化ヘルパ
    void normalize() {
        this->tag = this->upper(tag);
        this->isDiag = this->isDiagTag(tag);
        sort(qubits.begin(), qubits.end());
        qubits.erase(unique(qubits.begin(), qubits.end()), qubits.end());
    }
};