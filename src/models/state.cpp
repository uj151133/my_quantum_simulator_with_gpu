#include "state.hpp"
#include "../common/mathUtils.hpp"

/////////////////////////////////////
//
//	KET VECTORS
//
/////////////////////////////////////

QMDDSuite state::Ket0() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero},
    }));
};

QMDDSuite state::Ket1() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeZero},
        {edgeOne, edgeZero},
    }));
};

QMDDSuite state::KetPlus() {
    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeOne, edgeZero},
    }));
};

QMDDSuite state::KetMinus() {
    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {QMDDEdge(mathUtils::toLogPolar(-1.0)), edgeZero},
    }));
};

QMDDSuite state::KetI() {
    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {QMDDEdge(mathUtils::toLogPolar(i)), edgeZero},
    }));
};

QMDDSuite state::KetIMinus() {
    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {QMDDEdge(mathUtils::toLogPolar(-i)), edgeZero},
    }));
};


/////////////////////////////////////
//
//	BRA VECTORS
//
/////////////////////////////////////

QMDDSuite state::Bra0() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeZero}
    }));
};

QMDDSuite state::Bra1() {
    return QMDDSuite(mathUtils::toLogPolar(1.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeZero, edgeZero}
    }));
};

QMDDSuite state::BraPlus() {
    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeOne},
        {edgeZero, edgeZero}
    }));
};

QMDDSuite state::BraMinus() {
    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-1.0))},
        {edgeZero, edgeZero}
    }));
};

QMDDSuite state::BraI() {
    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(i))},
        {edgeZero, edgeZero}
    }));
};

QMDDSuite state::BraIMinus() {
    return QMDDSuite(mathUtils::toLogPolar(1.0 / M_SQRT2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(mathUtils::toLogPolar(-i))},
        {edgeZero, edgeZero}
    }));
};
