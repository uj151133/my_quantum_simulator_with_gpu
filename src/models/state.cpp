#include "state.hpp"

/////////////////////////////////////
//
//	KET VECTORS
//
/////////////////////////////////////

QMDDState state::Ket0() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne},
        {edgeZero},
    })));
};

QMDDState state::Ket1() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero},
        {edgeOne},
    })));
};

QMDDState state::KetPlus() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne},
        {edgeOne},
    })));
};

QMDDState state::KetMinus() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne},
        {QMDDEdge(-1.0, nullptr)},
    })));
};

QMDDState state::KetPlusY() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne},
        {QMDDEdge(i, nullptr)},
    })));
};

QMDDState state::KetMinusY() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne},
        {QMDDEdge(-i, nullptr)},
    })));
};


/////////////////////////////////////
//
//	BRA VECTORS
//
/////////////////////////////////////

QMDDState state::Bra0() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero}
    })));
};


QMDDState state::Bra1() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne}
    })));
};

QMDDState state::BraPlus() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeOne}
    })));
};

QMDDState state::BraMinus() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-1.0, nullptr)}
    })));
};

QMDDState state::BraPlusY() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(i, nullptr)}
    })));
};

QMDDState state::BraMinusY() {
    call_once(initEdgeFlag, initEdge);
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i, nullptr)}
    })));
};