#ifndef DAG_HPP
#define DAG_HPP
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <cctype>
#include "../common/Core.hpp"

using namespace std;

namespace dag {
// モデルが返す順列 perm を、可換性DAGに基づいて合法な順序に投影する
// 返り値: 合法な並べ替えインデックス列 order（size=N, 各要素は 0..N-1）
vector<int> tuneDAG(const vector<Core>& ops, const vector<int>& perm);

}

#endif