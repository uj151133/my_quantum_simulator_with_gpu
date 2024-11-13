package main

import (
    "fmt"
    "math/cmplx"
)

type QMDDNode struct {
    Edges [][]QMDDEdge
}

type QMDDEdge struct {
    Weight        complex128
    UniqueTableKey uint64
    IsTerminal    bool
}

func NewQMDDEdge(weight complex128, node *QMDDNode) QMDDEdge {
    var key uint64
    if node != nil {
        key = generateUniqueTableKey(*node)
    }
    return QMDDEdge{
        Weight:        weight,
        UniqueTableKey: key,
        IsTerminal:    node == nil,
    }
}

func NewQMDDEdgeWithComplex(w complex128, n *QMDDNode) *QMDDEdge {
    uniqueTableKey := uint64(0)
    isTerminal := n == nil
    if !isTerminal {
        uniqueTableKey = generateUniqueTableKey(n)
        table := GetUniqueTableInstance()
        if table.Find(uniqueTableKey) == nil {
            table.Insert(uniqueTableKey, n)
        }
    }
    return &QMDDEdge{weight: w, uniqueTableKey: uniqueTableKey, isTerminal: isTerminal}
}

func NewQMDDEdgeWithDouble(w float64, n *QMDDNode) *QMDDEdge {
    return NewQMDDEdgeWithComplex(complex(w, 0), n)
}

func NewQMDDEdgeWithComplexAndKey(w complex128, key uint64) *QMDDEdge {
    isTerminal := key == 0
    return &QMDDEdge{weight: w, uniqueTableKey: key, isTerminal: isTerminal}
}

func NewQMDDEdgeWithDoubleAndKey(w float64, key uint64) *QMDDEdge {
    return NewQMDDEdgeWithComplexAndKey(complex(w, 0), key)
}

func generateUniqueTableKey(node QMDDNode) uint64 {
    // ユニークなキーを生成するロジックをここに実装
    return 0 // 仮の値
}

func (edge QMDDEdge) GetStartNode() *QMDDNode {
    // UniqueTableからノードを取得するロジックをここに実装
    return nil // 仮の値
}

func (edge QMDDEdge) GetAllElementsForKet() []complex128 {
    var result []complex128
    var nodeStack []struct {
        Node      *QMDDNode
        EdgeIndex int
    }

    if edge.IsTerminal {
        result = append(result, edge.Weight)
    } else {
        nodeStack = append(nodeStack, struct {
            Node      *QMDDNode
            EdgeIndex int
        }{edge.GetStartNode(), 0})

        for len(nodeStack) > 0 {
            top := nodeStack[len(nodeStack)-1]
            nodeStack = nodeStack[:len(nodeStack)-1]

            if len(top.Node.Edges) == 1 {
                panic("The start node has only one edge, which is not allowed.")
            }

            for i := top.EdgeIndex; i < len(top.Node.Edges); i++ {
                if top.Node.Edges[i][0].IsTerminal {
                    result = append(result, top.Node.Edges[i][0].Weight)
                } else {
                    nodeStack = append(nodeStack, struct {
                        Node      *QMDDNode
                        EdgeIndex int
                    }{top.Node, i + 1})
                    nodeStack = append(nodeStack, struct {
                        Node      *QMDDNode
                        EdgeIndex int
                    }{top.Node.Edges[i][0].GetStartNode(), 0})
                    break
                }
            }
        }
    }
    return result
}

func (edge QMDDEdge) Equals(other QMDDEdge) bool {
    if edge.Weight != other.Weight {
        return false
    }
    if edge.IsTerminal != other.IsTerminal {
        return false
    }
    if !edge.IsTerminal && edge.UniqueTableKey != other.UniqueTableKey {
        return false
    }
    // UniqueTableからノードを比較するロジックをここに実装
    return true
}

func (edge QMDDEdge) NotEquals(other QMDDEdge) bool {
    return !edge.Equals(other)
}

func (edge QMDDEdge) String() string {
    return fmt.Sprintf("Weight = %v, Key = %v, isTerminal = %v", edge.Weight, edge.UniqueTableKey, edge.IsTerminal)
}

func NewQMDDNode(edges [][]QMDDEdge) *QMDDNode {
    return &QMDDNode{Edges: edges}
}

func (node *QMDDNode) Equals(other *QMDDNode) bool {
    if len(node.Edges) != len(other.Edges) {
        return false
    }
    for i := range node.Edges {
        if len(node.Edges[i]) != len(other.Edges[i]) {
            return false
        }
        for j := range node.Edges[i] {
            if !node.Edges[i][j].Equals(other.Edges[i][j]) {
                return false
            }
        }
    }
    return true
}

func (node *QMDDNode) NotEquals(other *QMDDNode) bool {
    return !node.Equals(other)
}

func (node *QMDDNode) String() string {
    result := fmt.Sprintf("Node with %d rows of edges\n", len(node.Edges))
    for i, row := range node.Edges {
        for j, edge := range row {
            result += fmt.Sprintf("    Edge (%d, %d): %s\n", i, j, edge.String())
        }
    }
    return result
}

type QMDDGate struct {
    InitialEdge QMDDEdge
}

func NewQMDDGate(edge QMDDEdge) *QMDDGate {
    return &QMDDGate{InitialEdge: edge}
}

func (gate *QMDDGate) GetStartNode() *QMDDNode {
    return gate.InitialEdge.GetStartNode()
}

func (gate *QMDDGate) GetInitialEdge() QMDDEdge {
    return gate.InitialEdge
}

func (gate *QMDDGate) Equals(other *QMDDGate) bool {
    return gate.InitialEdge.Equals(other.InitialEdge)
}

func (gate *QMDDGate) NotEquals(other *QMDDGate) bool {
    return !gate.Equals(other)
}

func (gate *QMDDGate) String() string {
    return fmt.Sprintf("Gate with initial edge: %s", gate.InitialEdge.String())
}

type QMDDState struct {
    InitialEdge QMDDEdge
}

func NewQMDDState(edge QMDDEdge) *QMDDState {
    return &QMDDState{InitialEdge: edge}
}

func (state *QMDDState) GetStartNode() *QMDDNode {
    return state.InitialEdge.GetStartNode()
}

func (state *QMDDState) GetInitialEdge() QMDDEdge {
    return state.InitialEdge
}

func (state *QMDDState) GetAllElements() []complex128 {
    return state.InitialEdge.GetAllElementsForKet()
}

func (state *QMDDState) Equals(other *QMDDState) bool {
    return state.InitialEdge.Equals(other.InitialEdge)
}

func (state *QMDDState) NotEquals(other *QMDDState) bool {
    return !state.Equals(other)
}

func (state *QMDDState) String() string {
    return fmt.Sprintf("State with initial edge: %s", state.InitialEdge.String())
}

