#ifndef CORE_HPP
#define CORE_HPP
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cctype>
#include <unordered_set>

using namespace std;

struct Core {
    string tag;           // ゲート種別（例: "H","CX","RZ","FUSED"...）
    vector<int> qubits;   // 対象量子ビット番号（1Q=1個, 2Q=2個）
    double theta = 0.0;        // 角度パラメータ1（回転系/位相系）
    double phi   = 0.0;        // 角度パラメータ2（必要に応じて使用）
    double lam   = 0.0;        // 角度パラメータ3（必要に応じて使用）
    double gamma = 0.0;        // 角度パラメータ4（必要に応じて使用）


    string shape = "GENERAL";

    uint64_t handle = 0;           // 融合ノード等のハンドル（任意）
    size_t edge_nodes = 0;    // コスト見積などに使う場合のノード数（任意）

    // 形状ラベル定数（大文字固定）
    static constexpr const char* kShapeDiag    = "DIAG";
    static constexpr const char* kShapeAnti    = "ANTI";
    static constexpr const char* kShapePerm    = "PERM";
    static constexpr const char* kShapeGeneral = "GENERAL";
    static constexpr const char* kShapeFused   = "FUSED";

    // 補助関数（定義は Core.cpp）
    static string upper(string s);
    static bool isDiagTag(const string& tagU);
    static string tagToShape(const string& tagU);
    static bool isSymmetric2QTag(const string& tagU);

    inline bool is2Q() const { return qubits.size()==2; }
    pair<int,int> orderedPair() const;
    pair<int,int> unorderedKey() const;
    void normalize();
};

#endif