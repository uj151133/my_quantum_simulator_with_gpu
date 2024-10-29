#include <metal_stdlib>
using namespace metal;

struct QMDDEdge {
    float2 weight;
    uint uniqueTableKey;
    bool isTerminal;
};

struct QMDDNode {
    device QMDDEdge* edges;
    uint edgeCount;
};

device QMDDNode* getNode(uint key, device QMDDNode* nodes, uint nodeCount) {
    for (uint i = 0; i < nodeCount; ++i) {
        if (nodes[i].uniqueTableKey == key) {
            return &nodes[i];
        }
    }
    return nullptr;
}

kernel void getAllElementsForKet(device QMDDEdge* edges [[buffer(0)]], device QMDDNode* nodes [[buffer(1)]], device float2* results [[buffer(2)]], uint nodeCount [[buffer(3)]], uint id [[thread_position_in_grid]]) {
    if (edges[id].isTerminal) {
        results[id] = edges[id].weight;
    } else {
        // スタックを使用して反復的な処理を行う
        uint stack[256];
        uint stackIndex = 0;
        stack[stackIndex++] = edges[id].uniqueTableKey;

        while (stackIndex > 0) {
            uint currentKey = stack[--stackIndex];
            device QMDDNode* currentNode = getNode(currentKey, nodes, nodeCount);

            if (currentNode == nullptr || currentNode->edgeCount == 1) {
                // エラー処理
                return;
            }

            for (uint i = 0; i < currentNode->edgeCount; ++i) {
                if (currentNode->edges[i].isTerminal) {
                    results[id] = currentNode->edges[i].weight;
                } else {
                    stack[stackIndex++] = currentNode->edges[i].uniqueTableKey;
                }
            }
        }
    }
}