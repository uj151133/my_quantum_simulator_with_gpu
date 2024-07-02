#ifndef QMDD_HPP
#define QMDD_HPP

#include <iostream>
#include <complex>
#include <array>
#include <vector>
#include <unordered_map>
#include <functional>

using namespace std; 

struct QMDDNode; // 前方宣言

struct QMDDEdge {
    complex<double> weight; // エッジの重み
    bool isTerminal; // 終端ノードかどうか
    QMDDNode* node; // エッジの指すQMDDNodeのポインタ

    QMDDEdge(complex<double> w = {0, 0}, QMDDNode* n = nullptr);
    bool operator==(const QMDDEdge& other) const;
};

struct QMDDNode {
    vector<QMDDEdge> edges; // エッジの配列
    size_t uniqueTableKey; // ユニークテーブルのキー

    QMDDNode(size_t numEdges = 2);
    bool operator==(const QMDDNode& other) const;
    void free();
};

class QMDDGate {
private:
    QMDDEdge initialEdge;

public:
    QMDDGate(QMDDEdge edge, size_t numEdges = 4);
    ~QMDDGate();
    QMDDNode* getStartNode() const;
    QMDDEdge getInitialEdge() const;
};

class QMDDState {
private:
    QMDDEdge initialEdge;

public:
    QMDDState(QMDDEdge edge, size_t numEdges = 2);

    ~QMDDState();

    QMDDNode* getStartNode() const;

    QMDDEdge getInitialEdge() const;

    // QMDDState同士の足し算
    QMDDState operator+(const QMDDState& other);

    QMDDNode* addNodes(QMDDNode* node1, QMDDNode* node2);
};

QMDDEdge mul(const QMDDEdge& m, const QMDDEdge& v);
QMDDEdge add(const QMDDEdge& e1, const QMDDEdge& e2);

#endif // QMDD_HPP
