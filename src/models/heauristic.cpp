#include "heauristic.hpp"

float heauristicCost(string_view shapeU) {
    if (shapeU == "DIAG") return ::COST_DIAG;
    if (shapeU == "ANTI") return ::COST_ANTI;
    if (shapeU == "PERM") return ::COST_PERM;
    return ::COST_GENERAL;
}

float heauristicScore(float cost) {
    return -cost;
}

vector<float> heauristicCosts(const vector<Core>& ops) {
    vector<float> out;
    out.reserve(ops.size());
    for (const auto& o : ops) {
        string s = o.shape;
        for (auto& c : s) c = (char)toupper((unsigned char)c);
        out.push_back(heauristicCost(s));
    }
    return out;
}

vector<float> heauristicScoresFromCosts(const vector<float>& costs) {
    vector<float> out;
    out.reserve(costs.size());
    for (float c : costs) out.push_back(heauristicScore(c));
    return out;
}