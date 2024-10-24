import Foundation
import Numerics

let i = Complex<Double>(0.0, 1.0)
let edgeOne = QMDDEdge(weight: 1.0, node: nil)
let edgeZero = QMDDEdge(weight: 0.0, node: nil)

// gate名前空間の定義
enum gate {
    static func I() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeOne, edgeZero],
            [edgeZero, edgeOne]
        ])))
    }

    static func Ph(delta: Double) -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: exp(i * delta), node: QMDDNode(edges: [
            [edgeOne, edgeZero],
            [edgeZero, edgeOne]
        ])))
    }

    static func X() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeZero, edgeOne],
            [edgeOne, edgeZero]
        ])))
    }
}