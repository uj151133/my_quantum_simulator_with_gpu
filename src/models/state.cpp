#include "state.hpp"

/////////////////////////////////////
//
//	KET VECTORS
//
/////////////////////////////////////

QMDDSuite state::Ket0() {
    return QMDDSuite(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero},
    })));
};

QMDDSuite state::Ket1() {
    return QMDDSuite(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero},
    })));
};

QMDDSuite state::KetPlus() {
    return QMDDSuite(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeOne, edgeZero},
    })));
};

QMDDSuite state::KetMinus() {
    return QMDDSuite(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {QMDDEdge(-1.0), edgeZero},
    })));
};

QMDDSuite state::KetI() {
    return QMDDSuite(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {QMDDEdge(i), edgeZero},
    })));
};

QMDDSuite state::KetIMinus() {
    return QMDDSuite(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {QMDDEdge(-i), edgeZero},
    })));
};


/////////////////////////////////////
//
//	BRA VECTORS
//
/////////////////////////////////////

QMDDSuite state::Bra0() {
    return QMDDSuite(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    })));
};

QMDDSuite state::Bra1() {
    return QMDDSuite(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    })));
};

QMDDSuite state::BraPlus() {
    return QMDDSuite(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeOne},
        {edgeZero, edgeZero}
    })));
};

QMDDSuite state::BraMinus() {
    return QMDDSuite(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-1.0)},
        {edgeZero, edgeZero}
    })));
};

QMDDSuite state::BraI() {
    return QMDDSuite(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(i)},
        {edgeZero, edgeZero}
    })));
};

QMDDSuite state::BraIMinus() {
    return QMDDSuite(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i)},
        {edgeZero, edgeZero}
    })));
};