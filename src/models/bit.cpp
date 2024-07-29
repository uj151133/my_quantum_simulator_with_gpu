#include "bit.hpp"

#include <ginac/ginac.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>



using namespace std;
using namespace GiNaC;

QMDDState state::KET_0() {
    auto ket0Node = make_shared<QMDDNode>(2);

    ket0Node->edges[0] = QMDDEdge(1.0, nullptr);
    ket0Node->edges[1] = QMDDEdge(0, nullptr);

    QMDDEdge ket0Edge(1.0, ket0Node);
    return QMDDState(ket0Edge);
};

QMDDState state::KET_1() {
    auto ket1Node = make_shared<QMDDNode>(2);

    ket1Node->edges[0] = QMDDEdge(0, nullptr);
    ket1Node->edges[1] = QMDDEdge(1.0, nullptr);

    QMDDEdge ket1Edge(1.0, ket1Node);
    return QMDDState(ket1Edge);
};

QMDDState state::KET_PLUS() {
    auto ketPlusNode = make_shared<QMDDNode>(2);

    ketPlusNode->edges[0] = QMDDEdge(sqrt(2.0), nullptr);
    ketPlusNode->edges[1] = QMDDEdge(sqrt(2.0), nullptr);

    QMDDEdge ketPlusEdge(1.0, ketPlusNode);
    return QMDDState(ketPlusEdge);
};

QMDDState state::KET_MINUS() {
    auto ketMinusNode = make_shared<QMDDNode>(2);

    ketMinusNode->edges[0] = QMDDEdge(sqrt(2.0), nullptr);
    ketMinusNode->edges[1] = QMDDEdge(-sqrt(2.0), nullptr);

    QMDDEdge ketMinusEdge(1.0, ketMinusNode);
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