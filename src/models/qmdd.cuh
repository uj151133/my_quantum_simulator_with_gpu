#include <cuComplex.h>

struct QMDDNode;  // 前方宣言

struct QMDDEdge {
    cuDoubleComplex weight;
    QMDDNode* node;

    // cuDoubleComplex 型のコンストラクタ
    QMDDEdge(cuDoubleComplex w, QMDDNode* n) : weight(w), node(n) {}

    // double 型のコンストラクタ
    QMDDEdge(double w, QMDDNode* n) : weight(make_cuDoubleComplex(w, 0.0)), node(n) {}
};
