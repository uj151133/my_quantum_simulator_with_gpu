#include "state.hpp"
/////////////////////////////////////
//
//	KET VECTORS
//
/////////////////////////////////////

QMDDState state::Ket0() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr)},
    })));
};

QMDDState state::Ket1() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr)},
    })));
};

QMDDState state::KetPlus() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr)},
        {QMDDEdge(1.0, nullptr)},
    })));
};

QMDDState state::KetMinus() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr)},
        {QMDDEdge(-1.0, nullptr)},
    })));
};

/////////////////////////////////////
//
//	BRA VECTORS
//
/////////////////////////////////////

QMDDState state::Bra0() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    })));
};


QMDDState state::Bra1() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    })));
};

QMDDState state::BraPlus() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(1.0, nullptr)}
    })));
};

QMDDState state::BraMinus() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-1.0, nullptr)}
    })));
};