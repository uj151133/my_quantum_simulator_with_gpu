#include "gate.hpp"



#include <iostream>

namespace py = pybind11;

complex<double> i(0, 1);

const QMDDGate gate::H_GATE = [] {
    complex<double> hWeight = 1.0 / sqrt(2.0);
    auto hNode = make_shared<QMDDNode>(4);

    hNode->edges[0] = QMDDEdge(1, nullptr);
    hNode->edges[1] = QMDDEdge(1, nullptr);
    hNode->edges[2] = QMDDEdge(1, nullptr);
    hNode->edges[3] = QMDDEdge(-1, nullptr);

    QMDDEdge hEdge(hWeight, hNode);
    return QMDDGate(hEdge);
}();

// const QMDDGate gate::H_GATE = [] {
//     complex<double> hWeight = 1.0 / sqrt(2.0);
//     auto hNode = std::make_unique<QMDDNode>(4);

//     // QMDDEdgeをムーブする
//     hNode->edges[0] = QMDDEdge(1, nullptr);
//     hNode->edges[1] = QMDDEdge(1, nullptr);
//     hNode->edges[2] = QMDDEdge(1, nullptr);
//     hNode->edges[3] = QMDDEdge(-1, nullptr);

//     // std::unique_ptrのリリースで生のポインタを渡す
//     QMDDEdge hEdge(hWeight, std::move(hNode));
//     return QMDDGate(std::move(hEdge));  // QMDDEdgeのムーブコンストラクタを使用
// }();


const QMDDGate gate::I_GATE = [] {
    complex<double> iWeight = 1.0;
    auto iNode = make_shared<QMDDNode>(4);

    iNode->edges[0] = QMDDEdge(1, nullptr);
    iNode->edges[1] = QMDDEdge(0, nullptr);
    iNode->edges[2] = QMDDEdge(0, nullptr);
    iNode->edges[3] = QMDDEdge(1, nullptr);

    QMDDEdge iEdge(iWeight, iNode);
    return QMDDGate(iEdge);
}();

const QMDDGate gate::X_GATE = [] {
    complex<double> xWeight = 1.0;
    auto xNode = make_shared<QMDDNode>(4);

    xNode->edges[0] = QMDDEdge(1, nullptr);
    xNode->edges[1] = QMDDEdge(0, nullptr);
    xNode->edges[2] = QMDDEdge(1, nullptr);
    xNode->edges[3] = QMDDEdge(0, nullptr);

    QMDDEdge xEdge(xWeight, xNode);
    return QMDDGate(xEdge);
}();



// QMDDGate createHGate() {
//     complex<double> hWeight = 1.0 / sqrt(2.0);
//     QMDDNode* hNode = new QMDDNode(4);

//     hNode->edges[0] = QMDDEdge(1, nullptr);
//     hNode->edges[1] = QMDDEdge(1, nullptr);
//     hNode->edges[2] = QMDDEdge(1, nullptr);
//     hNode->edges[3] = QMDDEdge(-1, nullptr);

//     QMDDEdge hEdge(hWeight, hNode);
//     return QMDDGate(hEdge);
// }

// QMDDGate createIGate() {
//     complex<double> iWeight = 1.0;
//     QMDDNode* iNode = new QMDDNode(4);

//     iNode->edges[0] = QMDDEdge(1, nullptr);
//     iNode->edges[1] = QMDDEdge(0, nullptr);
//     iNode->edges[2] = QMDDEdge(0, nullptr);
//     iNode->edges[3] = QMDDEdge(1, nullptr);

//     QMDDEdge iEdge(iWeight, iNode);
//     return QMDDGate(iEdge);
// }

// QMDDGate createXGate() {
//     complex<double> xWeight = 1.0;
//     QMDDNode* xNode = new QMDDNode(4);

//     xNode->edges[0] = QMDDEdge(1, nullptr);
//     xNode->edges[1] = QMDDEdge(0, nullptr);
//     xNode->edges[2] = QMDDEdge(1, nullptr);
//     xNode->edges[3] = QMDDEdge(0, nullptr);

//     QMDDEdge xEdge(xWeight, xNode);
//     return QMDDGate(xEdge);
// }

// QMDDGate createPlusXGate() {
//     complex<double> plusXWeight = 1 / sqrt(2.0);
//     QMDDNode* plusXNode = new QMDDNode(4);

//     plusXNode->edges[0] = QMDDEdge(1, nullptr);
//     plusXNode->edges[1] = QMDDEdge(i, nullptr);
//     plusXNode->edges[2] = QMDDEdge(i, nullptr);
//     plusXNode->edges[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusXEdge(plusXWeight, plusXNode);
//     return QMDDGate(plusXEdge);
// }

// QMDDGate createMinusXGate() {
//     complex<double> minusXWeight = 1 / sqrt(2.0);
//     QMDDNode* minusXNode = new QMDDNode(4);

//     minusXNode->edges[0] = QMDDEdge(1, nullptr);
//     minusXNode->edges[1] = QMDDEdge(-i, nullptr);
//     minusXNode->edges[2] = QMDDEdge(-i, nullptr);
//     minusXNode->edges[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusXEdge(minusXWeight, minusXNode);
//     return QMDDGate(minusXEdge);
// }

// QMDDGate createYGate() {
//     complex<double> yWeight = i;
//     QMDDNode* yNode = new QMDDNode(4);

//     yNode->edges[0] = QMDDEdge(0, nullptr);
//     yNode->edges[1] = QMDDEdge(-1, nullptr);
//     yNode->edges[2] = QMDDEdge(1, nullptr);
//     yNode->edges[3] = QMDDEdge(0, nullptr);

//     QMDDEdge yEdge(yWeight, yNode);
//     return QMDDGate(yEdge);
// }

// QMDDGate createPlusYGate() {
//     complex<double> plusYWeight = 1 / sqrt(2.0);
//     QMDDNode* plusYNode = new QMDDNode(4);

//     plusYNode->edges[0] = QMDDEdge(1, nullptr);
//     plusYNode->edges[1] = QMDDEdge(1, nullptr);
//     plusYNode->edges[2] = QMDDEdge(-1, nullptr);
//     plusYNode->edges[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusYEdge(plusYWeight, plusYNode);
//     return QMDDGate(plusYEdge);
// }

// QMDDGate createMinusYGate() {
//     complex<double> minusYWeight = 1 / sqrt(2.0);
//     QMDDNode* minusYNode = new QMDDNode(4);

//     minusYNode->edges[0] = QMDDEdge(1, nullptr);
//     minusYNode->edges[1] = QMDDEdge(-1, nullptr);
//     minusYNode->edges[2] = QMDDEdge(1, nullptr);
//     minusYNode->edges[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusYEdge(minusYWeight, minusYNode);
//     return QMDDGate(minusYEdge);
// }

// QMDDGate createZGate() {
//     complex<double> zWeight = 1.0;
//     QMDDNode* zNode = new QMDDNode(4);

//     zNode->edges[0] = QMDDEdge(1, nullptr);
//     zNode->edges[1] = QMDDEdge(0, nullptr);
//     zNode->edges[2] = QMDDEdge(0, nullptr);
//     zNode->edges[3] = QMDDEdge(-1, nullptr);

//     QMDDEdge zEdge(zWeight, zNode);
//     return QMDDGate(zEdge);
// }

// QMDDGate createSGate() {
//     complex<double> sWeight = 1.0;
//     QMDDNode* sNode = new QMDDNode(4);

//     sNode->edges[0] = QMDDEdge(1, nullptr);
//     sNode->edges[1] = QMDDEdge(0, nullptr);
//     sNode->edges[2] = QMDDEdge(0, nullptr);
//     sNode->edges[3] = QMDDEdge(i, nullptr);

//     QMDDEdge sEdge(sWeight, sNode);
//     return QMDDGate(sEdge);
// }

// QMDDGate createSDaggerGate() {
//     complex<double> sDaggerWeight = 1.0;
//     QMDDNode* sDaggerNode = new QMDDNode(4);

//     sDaggerNode->edges[0] = QMDDEdge(1, nullptr);
//     sDaggerNode->edges[1] = QMDDEdge(0, nullptr);
//     sDaggerNode->edges[2] = QMDDEdge(0, nullptr);
//     sDaggerNode->edges[3] = QMDDEdge(-i, nullptr);

//     QMDDEdge sDaggerEdge(sDaggerWeight, sDaggerNode);
//     return QMDDGate(sDaggerEdge);
// }

// QMDDGate createTGate() {
//     complex<double> tWeight = 1.0;
//     QMDDNode* tNode = new QMDDNode(4);

//     tNode->edges[0] = QMDDEdge(1, nullptr);
//     tNode->edges[1] = QMDDEdge(0, nullptr);
//     tNode->edges[2] = QMDDEdge(0, nullptr);
//     tNode->edges[3] = QMDDEdge(exp(i * complex<double>(M_PI / 4)), nullptr);

//     QMDDEdge tEdge(tWeight, tNode);
//     return QMDDGate(tEdge);
// }

// QMDDGate createTDaggerGate() {
//     complex<double> tDaggerWeight = 1.0;
//     QMDDNode* tDaggerNode = new QMDDNode(4);

//     tDaggerNode->edges[0] = QMDDEdge(1, nullptr);
//     tDaggerNode->edges[1] = QMDDEdge(0, nullptr);
//     tDaggerNode->edges[2] = QMDDEdge(0, nullptr);
//     tDaggerNode->edges[3] = QMDDEdge(-exp(i * complex<double>(M_PI / 4)), nullptr);

//     QMDDEdge tDaggerEdge(tDaggerWeight, tDaggerNode);
//     return QMDDGate(tDaggerEdge);
// }

// const matrix CNOT_GATE = matrix{
//     {1, 0, 0, 0}, 
//     {0, 1, 0, 0},
//     {0, 0, 0, 1},
//     {0, 0, 1, 0}
// };

// const matrix CZ_GATE = matrix{ 
//     {1, 0, 0, 0},
//     {0, 1, 0, 0},
//     {0, 0, 1, 0},
//     {0, 0, 0, -1}
// };

// const matrix TOFFOLI_GATE = matrix{ 
//     {1, 0, 0, 0, 0, 0, 0, 0},
//     {0, 1, 0, 0, 0, 0, 0, 0},
//     {0, 0, 1, 0, 0, 0, 0, 0},
//     {0, 0, 0, 1, 0, 0, 0, 0},
//     {0, 0, 0, 0, 1, 0, 0, 0},
//     {0, 0, 0, 0, 0, 1, 0, 0},
//     {0, 0, 0, 0, 0, 0, 0, 1},
//     {0, 0, 0, 0, 0, 0, 1, 0}
// };

// const matrix SWAP_GATE = matrix{ 
//     {1, 0, 0, 0},
//     {0, 0, 1, 0},
//     {0, 1, 0, 0}, 
//     {0, 0, 0, 1}
// };

// QMDDGate createRotateXGate(double theta) {
//     complex<double> rotateXWeight = 1.0;
//     QMDDNode* rotateXNode = new QMDDNode(4);

//     rotateXNode->edges[0] = QMDDEdge(cos(theta / 2), nullptr);
//     rotateXNode->edges[1] = QMDDEdge(-i * sin(theta / 2), nullptr);
//     rotateXNode->edges[2] = QMDDEdge(-i * sin(theta / 2), nullptr);
//     rotateXNode->edges[3] = QMDDEdge(cos(theta / 2), nullptr);

//     QMDDEdge rotateXEdge(rotateXWeight, rotateXNode);
//     return QMDDGate(rotateXEdge);
// }

// matrix RotateX(const ex &theta){
//     return matrix{
//         {cos(theta / 2), -I * sin(theta / 2)},
//         {-I * sin(theta / 2), cos(theta / 2)}
//     };
// }

// matrix RotateY(const ex &theta){
//     return matrix{ 
//         {cos(theta / 2), -sin(theta / 2)},
//         {sin(theta / 2), cos(theta / 2)}
//     };
// }

// matrix RotateZ(const ex &theta){
//     return matrix{
//         {exp(-I * theta / 2), 0},
//         {0, exp(I * theta / 2)}
//     };
// }
// matrix Rotate(const ex &k){
//     return matrix{
//         {1, 0},
//         {0, exp((2 * Pi * I) / pow(2, k))}
//     };
// }

// matrix U1(const ex &lambda){
//     return matrix{
//         {1, 0},
//         {0, exp(I * lambda)}
//     };
// }

// matrix U2(const ex &phi, const ex &lambda){
//     return matrix{
//         {1, -exp(I * lambda)},
//         {exp(I * phi), exp(I * (lambda + phi))}
//     };
// }

// matrix U3(const ex &theta, const ex &phi, const ex &lambda){
//     return matrix{
//         {cos(theta / 2), -exp(I * lambda) * sin(theta / 2)},
//         {exp(I * phi) * sin(theta / 2), exp(I * (lambda + phi)) * cos(theta / 2)}
//     };
// }

// // vector<vector<ex>> Ry(const ex &theta){
// //     return {
// //         {cos(theta / 2), -sin(theta / 2)},
// //         {sin(theta / 2), cos(theta / 2)}
// //         };
// // }

PYBIND11_MODULE(gate_py, m) {
    py::class_<QMDDNode, std::shared_ptr<QMDDNode>>(m, "QMDDNode")
        .def(py::init<int>());

    py::class_<QMDDEdge>(m, "QMDDEdge")
        .def(py::init<std::complex<double>, std::shared_ptr<QMDDNode>>());

    py::class_<QMDDGate>(m, "QMDDGate")
        .def_static("get_h_gate", []() { return gate::H_GATE; }, py::return_value_policy::reference);

    m.attr("H_GATE") = gate::H_GATE;
}