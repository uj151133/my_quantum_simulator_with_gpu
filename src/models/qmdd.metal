#include <metal_stdlib>
using namespace metal;

struct QMDDNode {
    device QMDDNode* edges;
};

struct QMDDEdge {
    double2 weight;
    uint uniqueTableKey;
    bool isTerminal;
};

kernel void processQMDDNodes(device QMDDNode* nodes [[buffer(0)]],
                             device QMDDEdge* edges [[buffer(1)]],
                             uint nodeCount [[thread_position_in_grid]]) {
    if (nodeCount >= nodes->edges->uniqueTableKey) return;

    QMDDNode node = nodes[nodeCount];
    for (uint i = 0; i < node.edges->uniqueTableKey; i++) {
        QMDDEdge edge = edges[i];
        // ここでエッジの処理を行う
    }
}