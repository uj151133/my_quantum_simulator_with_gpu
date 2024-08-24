#include "bit.hpp"
/////////////////////////////////////
//
//	KET VECTORS
//
/////////////////////////////////////

QMDDState state::KET_0() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr)},
        {QMDDEdge(.0, nullptr)},
    })));
};

QMDDState state::KET_1() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr)},
        {QMDDEdge(1.0, nullptr)},
    })));
};

QMDDState state::KET_PLUS() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr)},
        {QMDDEdge(1.0, nullptr)},
    })));
};

QMDDState state::KET_MINUS() {
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

QMDDState state::BRA_0() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    })));
};

QMDDState state::BRA_1() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    })));
};

QMDDState state::BRA_PLUS() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(1.0, nullptr)}
    })));
};

QMDDState state::BRA_MINUS() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-1.0, nullptr)}
    })));
};