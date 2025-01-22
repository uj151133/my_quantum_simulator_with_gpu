import Foundation
import Numerics

enum state {

    /////////////////////////////////////
    //
    //	KET VECTORS
    //
    /////////////////////////////////////

    static func Ket0() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeOne],
            [edgeZero]
        ])))
    }

    static func Ket1() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeZero],
            [edgeOne]
        ])))
    }

    static func KetPlus() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0 / sqrt(2.0), node: QMDDNode(edges: [
            [edgeOne],
            [edgeOne]
        ])))
    }

    static func KetMinus() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0 / sqrt(2.0), node: QMDDNode(edges: [
            [edgeOne],
            [QMDDEdge(weight: -1.0, node: nil)]
        ])))
    }

    static func KetPlusY() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0 / sqrt(2.0), node: QMDDNode(edges: [
            [edgeOne],
            [QMDDEdge(weight: i, node: nil)]
        ])))
    }

    static func KetMinusY() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0 / sqrt(2.0), node: QMDDNode(edges: [
            [edgeOne],
            [QMDDEdge(weight: -i, node: nil)]
        ])))
    }

    /////////////////////////////////////
    //
    //	BRA VECTORS
    //
    /////////////////////////////////////

    static func Bra0() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeOne, edgeZero]
        ])))
    }

    static func Bra1() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0, node: QMDDNode(edges: [
            [edgeZero, edgeOne]
        ])))
    }

    static func BraPlus() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0 / sqrt(2.0), node: QMDDNode(edges: [
            [edgeOne, edgeOne]
        ])))
    }

    static func BraMinus() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0 / sqrt(2.0), node: QMDDNode(edges: [
            [edgeOne, QMDDEdge(weight: -1.0, node: nil)]
        ])))
    }

    static func BraPlusY() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0 / sqrt(2.0), node: QMDDNode(edges: [
            [edgeOne, QMDDEdge(weight: i, node: nil)]
        ])))
    }

    static func BraMinusY() -> QMDDState {
        return QMDDState(edge: QMDDEdge(weight: 1.0 / sqrt(2.0), node: QMDDNode(edges: [
            [edgeOne, QMDDEdge(weight: -i, node: nil)]
        ])))
    }
}