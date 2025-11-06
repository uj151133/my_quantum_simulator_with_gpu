#include "dag.hpp"


namespace {

static inline string upper(string s){
    for(auto& c: s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

// 対角タグ（shape情報が無い場合のフォールバック）
static inline bool is_diag_tag(const string& u){
    return u=="RZ"||u=="U1"||u=="P"||u=="S"||u=="T"||u=="Z"||
           u=="CZ"||u=="CP"||u=="CRZ"||u=="RZZ";
}

static inline bool is_diag(const Core& o){
    if (!o.shape.empty() && Core::upper(o.shape)==Core::kShapeDiag) return true;
    return is_diag_tag(Core::upper(o.tag));
}

// 「同一control・別target」の CX/CNOT 同士は依存なし
static inline bool cx_same_ctrl_diff_tgt(const Core& a, const Core& b){
    auto ua = Core::upper(a.tag), ub = Core::upper(b.tag);
    if(!((ua=="CX"||ua=="CNOT") && (ub=="CX"||ub=="CNOT"))) return false;
    if(a.qubits.size()!=2 || b.qubits.size()!=2) return false;
    return a.qubits[0]==b.qubits[0] && a.qubits[1]!=b.qubits[1];
}

} // namespace

namespace dag {

vector<int> tuneDAG(const vector<Core>& ops, const vector<int>& perm){
    const int N = static_cast<int>(ops.size());
    if (static_cast<int>(perm.size()) != N) return {};

    // perm 内の順位 rank[v]（小さいほど優先）
    vector<int> rank(N, N+N);
    for(int i=0;i<N;i++){
        int v = perm[i];
        if (v>=0 && v<N) rank[v] = i;
    }

    // 依存グラフ: 「同じ量子ビットの直前タッチ」にのみ辺を張る（必要最小限）
    vector<unordered_set<int>> preds(N), succs(N);
    unordered_map<int,int> last_touch;
    for(int i=0;i<N;i++){
        const auto& op = ops[i];
        for(int q : op.qubits){
            auto it = last_touch.find(q);
            if (it != last_touch.end()){
                int j = it->second;
                const auto& a = ops[j];
                const auto& b = ops[i];
                const bool both_diag = is_diag(a) && is_diag(b);
                if (!(both_diag || cx_same_ctrl_diff_tgt(a,b))){
                    preds[i].insert(j);
                    succs[j].insert(i);
                }
            }
            last_touch[q] = i;
        }
    }

    // ready 集合（入次数0）
    set<int> ready;
    for(int i=0;i<N;i++) if (preds[i].empty()) ready.insert(i);

    // perm に最も近い合法トポロジカル順へ投影
    vector<int> order; order.reserve(N);
    while(!ready.empty()){
        // ready 中で rank が最小のノードを選ぶ
        int pick = *min_element(ready.begin(), ready.end(),
            [&](int a, int b){ return rank[a] < rank[b]; });
        ready.erase(pick);
        order.push_back(pick);
        // 後続の入次数更新
        for(int v : succs[pick]){
            preds[v].erase(pick);
            if (preds[v].empty()) ready.insert(v);
        }
    }
    return order;
}

}