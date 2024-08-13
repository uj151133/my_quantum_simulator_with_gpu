#include "bit.hpp"

#include <ginac/ginac.h>


using namespace std;
using namespace GiNaC;

QMDDState state::KET_0() {

    vector<QMDDEdge> ket0Children = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(0.0, nullptr)
    };

    auto ket0Node = make_shared<QMDDNode>(ket0Children);

    QMDDEdge ket0Edge(1.0, ket0Node);
    return QMDDState(ket0Edge);
};

QMDDState state::KET_1() {

    vector<QMDDEdge> ket1Children = {
        QMDDEdge(0.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto ket1Node = make_shared<QMDDNode>(ket1Children);

    QMDDEdge ket1Edge(1.0, ket1Node);
    return QMDDState(ket1Edge);
};

QMDDState state::KET_PLUS() {

    vector<QMDDEdge> ketPlusChildren = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(1.0, nullptr)
    };

    auto ketPlusNode = make_shared<QMDDNode>(ketPlusChildren);

    QMDDEdge ketPlusEdge(1.0 / sqrt(2.0), ketPlusNode);
    return QMDDState(ketPlusEdge);
};

QMDDState state::KET_MINUS() {

    vector<QMDDEdge> ketMinusChildren = {
        QMDDEdge(1.0, nullptr),
        QMDDEdge(-1.0, nullptr)
    };

    auto ketMinusNode = make_shared<QMDDNode>(ketMinusChildren);

    QMDDEdge ketMinusEdge(1.0 / sqrt(2.0), ketMinusNode);
    return QMDDState(ketMinusEdge);
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