#ifndef QMDD_HPP
#define QMDD_HPP

#include <iostream>
#include <complex>
#include <array>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cmath> 
#include <memory>

using namespace std; 

struct QMDDNode; // 前方宣言

struct QMDDEdge {
    complex<double> weight; // エッジの重み
    bool isTerminal; // 終端ノードかどうか
    // unique_ptr<QMDDNode> node; // エッジの指すQMDDNodeのポインタ
    QMDDNode* node;
    
    QMDDEdge(complex<double> w = {0, 0}, QMDDNode* n = nullptr);
    ~QMDDEdge();
    bool operator==(const QMDDEdge& other) const;
    friend ostream& operator<<(ostream& os, const QMDDEdge& edge);
};

struct QMDDNode {
    vector<QMDDEdge> edges; // エッジの配列
    size_t uniqueTableKey; // ユニークテーブルのキー

    QMDDNode(size_t numEdges = 2);
    ~QMDDNode();
    QMDDNode(const QMDDNode&);
    QMDDNode& operator=(const QMDDNode&);
    QMDDNode(QMDDNode&& other) noexcept; // ムーブコンストラクタ
    QMDDNode& operator=(QMDDNode&& other) noexcept; // ムーブ代入演算子
    bool operator==(const QMDDNode& other) const;
    friend ostream& operator<<(ostream& os, const QMDDNode& node);
};

class QMDDGate {
private:
    QMDDEdge initialEdge;

public:
    QMDDGate(QMDDEdge edge, size_t numEdges = 4);
    ~QMDDGate();
    QMDDNode* getStartNode() const;
    QMDDEdge getInitialEdge() const;
    friend ostream& operator<<(ostream& os, const QMDDGate& gate);
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
    friend ostream& operator<<(ostream& os, const QMDDState& state);
};
#endif // QMDD_HPP
