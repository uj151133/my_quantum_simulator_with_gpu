#ifndef QMDD_HPP
#define QMDD_HPP

#include <variant>
#include <iostream>
#include <complex>
#include <array>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <memory>

using namespace std;

struct QMDDNode;

struct QMDDEdge;

class QMDDGate;

class QMDDState;


using QMDDVariant = variant<QMDDGate, QMDDState>;
ostream& operator<<(ostream& os, const QMDDVariant& variant);


class QMDDNodeHashHelper {
public:
    size_t calculateMatrixHash(const QMDDNode& node) const;
private:
    size_t calculateMatrixHash(const QMDDNode& node, size_t row, size_t col, size_t rowStride, size_t colStride, const complex<double>& parentWeight) const;
    size_t hashMatrixElement(const complex<double>& value, size_t row, size_t col) const;
    size_t customHash(const complex<double>& value) const;
};



struct QMDDEdge{
    complex<double> weight; // エッジの重み
    size_t uniqueTableKey;
    bool isTerminal; // 終端ノードかどうか
    shared_ptr<QMDDNode> node; // エッジの指すQMDDNodeのポインタ

    QMDDEdge(complex<double> w = {0.0, 0.0}, shared_ptr<QMDDNode> n = nullptr);
    QMDDEdge(double w, shared_ptr<QMDDNode> n);
    QMDDEdge(const QMDDEdge& other) = default;
    ~QMDDEdge() = default;
    QMDDEdge& operator=(const QMDDEdge& other) = default;
    bool operator==(const QMDDEdge& other) const;
    bool operator!=(const QMDDEdge& other) const;
    friend ostream& operator<<(ostream& os, const QMDDEdge& edge);
};

struct QMDDNode {
    vector<QMDDEdge> edges; // エッジの配列

    QMDDNode(const vector<QMDDEdge>& edges);
    ~QMDDNode() = default;
    // コピーコンストラクタとコピー代入演算子
    QMDDNode(const QMDDNode& other) = default;
    QMDDNode& operator=(const QMDDNode& other) = default;
    // ムーブコンストラクタとムーブ代入演算子
    QMDDNode(QMDDNode&& other) noexcept = default;
    QMDDNode& operator=(QMDDNode&& other) noexcept;
    bool operator==(const QMDDNode& other) const;
    bool operator!=(const QMDDNode& other) const;
    friend ostream& operator<<(ostream& os, const QMDDNode& node);
};

class QMDDGate{
private:
    QMDDEdge initialEdge;
    size_t depth;
public:
    QMDDGate(QMDDEdge edge, size_t numEdge = 4);
    QMDDGate(const QMDDGate& other) = default;
    ~QMDDGate() = default;
    QMDDNode* getStartNode() const;
    QMDDEdge getInitialEdge() const;
    size_t getDepth() const;
    void calculateDepth();
    QMDDGate& operator=(const QMDDGate& other) = default;
    bool operator==(const QMDDGate& other) const;
    bool operator!=(const QMDDGate& other) const;
    friend ostream& operator<<(ostream& os, const QMDDGate& gate);
};

class QMDDState{
private:
    QMDDEdge initialEdge;

public:
    QMDDState(QMDDEdge edge, size_t numEdge = 2);
    QMDDState(const QMDDState& other) = default;
    ~QMDDState() = default;
    QMDDNode* getStartNode() const;
    QMDDEdge getInitialEdge() const;
    QMDDState operator+(const QMDDState& other);
    shared_ptr<QMDDNode> addNodes(QMDDNode* node1, QMDDNode* node2);
    QMDDState& operator=(const QMDDState& other) = default;
    bool operator==(const QMDDState& other) const;
    bool operator!=(const QMDDState& other) const;
    friend ostream& operator<<(ostream& os, const QMDDState& state);
};
#endif // QMDD_HPP
