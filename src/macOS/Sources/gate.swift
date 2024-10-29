import Foundation
import Numerics

enum gate {
    static func I() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeOne, edgeZero],
            [edgeZero, edgeOne]
        ])))
    }

    static func Ph(delta: Double) -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: Complex<Double>.exp(i * Complex<Double>(delta, 0.0)), node: QMDDNode(edges: [
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
        let vEdge = QMDDEdge(weight: i, node: nil)

        return QMDDGate(edge: QMDDEdge(weight: Complex<Double>(1.0 / 2.0, 1.0 / 2.0), node: QMDDNode(edges: [
            [edgeOne, vEdge],
            [vEdge, edgeOne]
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
        let cx2Edge1 = QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeOne, edgeZero],
            [edgeZero, edgeZero]
        ]))

        let cx2Edge2 = QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeZero, edgeZero],
            [edgeZero, edgeOne]
        ]))

        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [cx2Edge1, cx2Edge2],
            [cx2Edge2, cx2Edge1]
        ])))
    }

    static func varCX() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [gate.X().getInitialEdge(), edgeZero],
            [edgeZero, gate.I().getInitialEdge()]
        ])))
    }

    static func CZ() -> QMDDGate {
        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [gate.I().getInitialEdge(), edgeZero],
            [edgeZero, gate.Z().getInitialEdge()]
        ])))
    }

        static func DCNOT() -> QMDDGate {
        let dcnotEdge1 = QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeOne, edgeZero],
            [edgeZero, edgeZero]
        ]))

        let dcnotEdge2 = QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeZero, edgeZero],
            [edgeOne, edgeZero]
        ]))

        let dcnotEdge3 = QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeZero, edgeZero],
            [edgeZero, edgeOne]
        ]))

        let dcnotEdge4 = QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeZero, edgeOne],
            [edgeZero, edgeZero]
        ]))

        return QMDDGate(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [dcnotEdge1, dcnotEdge2],
            [dcnotEdge3, dcnotEdge4]
        ])))
    }
}