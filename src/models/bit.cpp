#include "bit.hpp"

#include <ginac/ginac.h>


using namespace std;
using namespace GiNaC;

QMDDState state::KET_0() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(.0, nullptr),
    })));
};

QMDDState state::KET_1() {
    return QMDDState(QMDDEdge(1.0, make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(.0, nullptr),
        QMDDEdge(1.0, nullptr),
    })));
};

QMDDState state::KET_PLUS() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(1.0, nullptr),
    })));
};

QMDDState state::KET_MINUS() {
    return QMDDState(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<QMDDEdge>{
        QMDDEdge(1.0, nullptr),
        QMDDEdge(-1.0, nullptr),
    })));
};

// const matrix BRA_0 = matrix{
// {1, 0}
// };

// const matrix BRA_1 = matrix{
// {0, 1}
// };

// const matrix BRA_PLUS = matrix{
// {sqrt(ex(2)), sqrt(ex(2))}
// };

// const matrix BRA_MINUS = matrix{
// {sqrt(ex(2)), -sqrt(ex(2))}
// };

// PYBIND11_MODULE(bit, m) {

//     m.attr("KET_0") = KET_0;
//     m.attr("KET_1") = KET_1;
//     m.attr("KET_PLUS") = KET_PLUS;
//     m.attr("KET_MINUS") = KET_MINUS;
//     m.attr("BRA_0") = BRA_0;
//     m.attr("BRA_1") = BRA_1;
//     m.attr("BRA_PLUS") = BRA_PLUS;
//     m.attr("BRA_MINUS") = BRA_MINUS;
// }