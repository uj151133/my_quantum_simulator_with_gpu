import Metal
import Foundation
import Numerics

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

let library = device.makeDefaultLibrary()!
let kernelFunction = library.makeFunction(name: "getAllElementsForKet")!
let pipelineState = try! device.makeComputePipelineState(function: kernelFunction)


enum OperationType {
    case add
    case mul
    case kronecker
}

typealias OperationKey = (QMDDEdge, OperationType, QMDDEdge)

typealias OperationResult = (Complex<Double>, UInt32)

enum QMDDVariant {
    case gate(QMDDGate)
    case state(QMDDState)
}

extension QMDDVariant: CustomStringConvertible {
    var description: String {
        switch self {
        case .gate(let gate):
            return "\(gate)"
        case .state(let state):
            return "\(state)"
        }
    }
}


/////////////////////////////////////
//
//	QMDDEdge
//
/////////////////////////////////////

struct QMDDEdge: CustomStringConvertible {
    var weight: Complex<Double>
    var uniqueTableKey: UInt32
    var isTerminal: Bool

    init(weight: Complex<Double> = Complex<Double>(0.0, 0.0), node: QMDDNode? = nil) {
        self.weight = weight
        self.uniqueTableKey = node != nil ? calculation.generateUniqueTableKey(node!) : 0
        self.isTerminal = node == nil
        if let node = node {
            let table = UniqueTable.getInstance()
            if table.find(uniqueTableKey) == nil {
                table.insert(uniqueTableKey, node)
            }
        }
    }

    init(weight: Double, node: QMDDNode? = nil) {
        self.weight = Complex<Double>(weight, 0.0)
        self.uniqueTableKey = node != nil ? calculation.generateUniqueTableKey(node!) : 0
        self.isTerminal = node == nil
        if let node = node {
            let table = UniqueTable.getInstance()
            if table.find(uniqueTableKey) == nil {
                table.insert(uniqueTableKey, node)
            }
        }
    }

    init(weight: Complex<Double> = Complex<Double>(0.0, 0.0), uniqueTableKey: UInt32 = 0, isTerminal: Bool = false) {
        self.weight = weight
        self.uniqueTableKey = uniqueTableKey
        self.isTerminal = isTerminal
    }

    init(weight: Double, uniqueTableKey: UInt32 = 0, isTerminal: Bool = false) {
        self.weight = Complex<Double>(weight, 0.0)
        self.uniqueTableKey = uniqueTableKey
        self.isTerminal = isTerminal
    }

    func getStartNode() -> QMDDNode? {
        let table = UniqueTable.getInstance()
        return table.find(uniqueTableKey)
    }

    func getAllElementsForKet() -> [Complex<Double>] {
        var result: [Complex<Double>] = []
        var nodeStack: [(QMDDNode, Int)] = []

        if isTerminal {
            result.append(weight)
        } else {
            if let startNode = getStartNode() {
                nodeStack.append((startNode, 0))
            }

            while !nodeStack.isEmpty {
                let (node, edgeIndex) = nodeStack.removeLast()

                if node.edges.count == 1 {
                    fatalError("The start node has only one edge, which is not allowed.")
                }

                for i in edgeIndex..<node.edges.count {
                    if node.edges[i][0].isTerminal {
                        result.append(node.edges[i][0].weight)
                    } else {
                        nodeStack.append((node, i + 1))
                        if let nextNode = node.edges[i][0].getStartNode() {
                            nodeStack.append((nextNode, 0))
                        }
                        break
                    }
                }
            }
        }
        return result
    }

    static func ==(lhs: QMDDEdge, rhs: QMDDEdge) -> Bool {
        let table = UniqueTable.getInstance()
        if lhs.weight != rhs.weight { return false }
        if lhs.isTerminal != rhs.isTerminal { return false }
        if !lhs.isTerminal && lhs.uniqueTableKey != rhs.uniqueTableKey { return false }
        if !lhs.isTerminal && table.find(lhs.uniqueTableKey) != table.find(rhs.uniqueTableKey) { return false }
        return true
    }

    static func !=(lhs: QMDDEdge, rhs: QMDDEdge) -> Bool {
        return !(lhs == rhs)
    }

    var description: String {
        var desc = "Weight: \(weight), Node"
        if uniqueTableKey != 0 {
            desc += ", Key: \(uniqueTableKey)"
        } else {
            desc += ", Key: Null"
        }
        return desc
    }
}

/////////////////////////////////////
//
//	QMDDNode
//
/////////////////////////////////////

struct QMDDNode: CustomStringConvertible {
    var edges: [[QMDDEdge]]

    init(edges: [[QMDDEdge]]) {
        self.edges = edges
    }

    mutating func moveEdges(from other: QMDDNode) {
        self.edges = other.edges
        other.edges.removeAll()
    }

    static func ==(lhs: QMDDNode, rhs: QMDDNode) -> Bool {
        if lhs.edges.count != rhs.edges.count {
            return false
        }
        for i in 0..<lhs.edges.count {
            if lhs.edges[i].count != rhs.edges[i].count {
                return false
            }
            for j in 0..<lhs.edges[i].count {
                if lhs.edges[i][j] != rhs.edges[i][j] {
                    return false
                }
            }
        }
        return true
    }

    static func !=(lhs: QMDDNode, rhs: QMDDNode) -> Bool {
        return !(lhs == rhs)
    }

    var description: String {
        var desc = "QMDDNode with \(edges.count) rows of edges \n"
        for row in edges {
            desc += "Row with \(row.count) edges: "
            for edge in row {
                desc += "\(edge) "
            }
            desc += "\n"
        }
        return desc
    }
}

/////////////////////////////////////
//
//	QMDDGate
//
/////////////////////////////////////

class QMDDGate: CustomStringConvertible {
    private var initialEdge: QMDDEdge
    private var depth: Int

    init(edge: QMDDEdge, numEdge: Int = 4) {
        self.initialEdge = edge
        self.depth = 0
        calculateDepth()
    }

    func calculateDepth() {
        guard var currentNode = getStartNode() else {
            return
        }
        var currentDepth = 0

        while !currentNode.edges.isEmpty {
            currentDepth += 1
            if let nextNode = currentNode.edges[0][0].getStartNode() {
                currentNode = nextNode
            } else {
                break
            }
        }
        depth = currentDepth
    }

    func getStartNode() -> QMDDNode? {
        let table = UniqueTable.getInstance()
        return table.find(initialEdge.uniqueTableKey)
    }

    func getInitialEdge() -> QMDDEdge {
        return initialEdge
    }

    func getDepth() -> Int {
        return depth
    }

    static func ==(lhs: QMDDGate, rhs: QMDDGate) -> Bool {
        return lhs.initialEdge == rhs.initialEdge && lhs.depth == rhs.depth
    }

    static func !=(lhs: QMDDGate, rhs: QMDDGate) -> Bool {
        return !(lhs == rhs)
    }

    var description: String {
        return "QMDDGate with initial edge:\n\(initialEdge), depth: \(depth)"
    }
}

/////////////////////////////////////
//
//	QMDDState
//
/////////////////////////////////////

class QMDDState: CustomStringConvertible {
    private var initialEdge: QMDDEdge
    private var depth: Int

    init(edge: QMDDEdge) {
        self.initialEdge = edge
        self.depth = 0
        calculateDepth()
    }

    func calculateDepth() {
        guard var currentNode = getStartNode() else {
            return
        }
        var currentDepth = 0

        while !currentNode.edges.isEmpty {
            currentDepth += 1
            if let nextNode = currentNode.edges[0][0].getStartNode() {
                currentNode = nextNode
            } else {
                break
            }
        }
        depth = currentDepth
    }

    func getStartNode() -> QMDDNode? {
        let table = UniqueTable.getInstance()
        return table.find(initialEdge.uniqueTableKey)
    }

    func getInitialEdge() -> QMDDEdge {
        return initialEdge
    }

    func getDepth() -> Int {
        return depth
    }

    static func ==(lhs: QMDDState, rhs: QMDDState) -> Bool {
        return lhs.initialEdge == rhs.initialEdge
    }

    static func !=(lhs: QMDDState, rhs: QMDDState) -> Bool {
        return !(lhs == rhs)
    }

    var description: String {
        return "QMDDState with initial edge:\n\(initialEdge)"
    }
}

func compareAndSwap<T: Equatable>(_ variable: inout T, expected: T, newValue: T) -> Bool {
    if variable == expected {
        variable = newValue
        return true  // 成功
    }
    return false  // 失敗
}