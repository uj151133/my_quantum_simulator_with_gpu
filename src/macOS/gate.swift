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

    static func Y() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: -i, node: QMDDNode(edges: [
            [edgeZero, edgeOne],
            [QMDDEdge(weight: -1.0, node: nil), edgeZero]
        ])))
    }

    static func Z() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeOne, edgeZero],
            [edgeZero, QMDDEdge(weight: -1.0, node: nil)]
        ])))
    }

    static func S() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeOne, edgeZero],
            [edgeZero, QMDDEdge(weight: i, node: nil)]
        ])))
    }

    static func Sdagger() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeOne, edgeZero],
            [edgeZero, QMDDEdge(weight: -i, node: nil)]
        ])))
    }

    static func V() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0 / 2.0 + i / 2.0, node: QMDDNode(edges: [
            [edgeOne, QMDDEdge(weight: i, node: nil)],
            [QMDDEdge(weight: i, node: nil), QMDDEdge(weight: 0.5, node: nil)]
        ])))
    }

    static func H() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0 / sqrt(2.0), node: QMDDNode(edges: [
            [edgeOne, edgeOne],
            [edgeOne, QMDDEdge(weight: -1.0, node: nil)]
        ])))
    }

    static func CX1() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [gate.I().getInitialEdge(), edgeZero],
            [edgeZero, gate.X().getInitialEdge()]
        ])))
    }

    static func CX2() -> QMDDGate {
        let cx2Node1 = QMDDNode(edges: [
            [edgeOne, edgeZero],
            [edgeZero, edgeZero]
        ])

        let cx2Node2 = QMDDNode(edges: [
            [edgeZero, edgeZero],
            [edgeZero, edgeOne]
        ])

        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [QMDDEdge(weight: 1.0, node: cx2Node1), QMDDEdge(weight: 1.0, node: cx2Node2)],
            [QMDDEdge(weight: 1.0, node: cx2Node2), QMDDEdge(weight: 1.0, node: cx2Node1)]
        ])))
    }

    static func varCX() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [gate.X().getInitialEdge(), edgeZero],
            [edgeZero, gate.I().getInitialEdge()]
        ])))
    }
}