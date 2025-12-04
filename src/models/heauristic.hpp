#ifndef HEAURISTIC_HPP
#define HEAURISTIC_HPP

#include <vector>
#include <string_view>
#include "../common/Core.hpp"

using namespace std;

inline constexpr float COST_DIAG    = 2.0f;
inline constexpr float COST_ANTI    = 2.5f;
inline constexpr float COST_PERM    = 3.0f;
inline constexpr float COST_GENERAL = 4.0f;


float heauristicCost(string_view shapeU);
float heauristicScore(float cost);

vector<float> heauristicCosts(const vector<Core>& ops);
vector<float> heauristicScoresFromCosts(const vector<float>& costs);
inline vector<float> heauristicScores(const vector<Core>& ops) {
    return heauristicScoresFromCosts(heauristicCosts(ops));
}
#endif