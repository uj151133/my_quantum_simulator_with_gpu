#include "state.hpp"

/////////////////////////////////////
//
//	KET VECTORS
//
/////////////////////////////////////

QMDDState state::Ket0() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne},
        {edgeZero},
    })));
};

QMDDState state::Ket1() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero},
        {edgeOne},
    })));
};

QMDDState state::KetPlus() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne},
        {edgeOne},
    })));
};

QMDDState state::KetMinus() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne},
        {QMDDEdge(-1.0, nullptr)},
    })));
};

QMDDState state::KetPlusY() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne},
        {QMDDEdge(i, nullptr)},
    })));
};

QMDDState state::KetMinusY() {
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
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero}
    })));
};


QMDDState state::Bra1() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne}
    })));
};

QMDDState state::BraPlus() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeOne}
    })));
};

QMDDState state::BraMinus() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-1.0, nullptr)}
    })));
};

QMDDState state::BraPlusY() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(i, nullptr)}
    })));
};

QMDDState state::BraMinusY() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i, nullptr)}
    })));
};