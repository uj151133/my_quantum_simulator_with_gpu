#include "iostream"
#include <ginac/ginac.h>
#include "src/models/bit.hpp"
#include "src/models/gate.hpp"
#include "src/models/uniquetable.hpp"
#include "src/models/qmdd.hpp"
#include "src/common/calculation.hpp"

using namespace GiNaC;

int main() {
    // auto [hNode, hWeight] = createHGate();
    QMDDGate hGate(createHGate());

    // H_gate の確認
    cout << "hGate initial weight: " << hGate.getInitialEdge().weight << endl;
    // cout << "hWeight: " << hWeight << endl;
    for (const auto& edge : hGate.getStartNode()->edges) {
        cout << "hGate edge weight: " << edge.weight << endl;
    }

    // for (const auto& edge : hNode->edges) {
    //     cout << "hNode weight: " << edge.isTerminal << endl;
    // }

    return 0;
}