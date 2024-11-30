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
#include <tuple>
#include <stack>

using namespace std;

struct QMDDNode;

struct QMDDEdge;

class QMDDGate;

class QMDDState;


enum class OperationType {
    ADD,
    MUL,
    KRONECKER,
};

using OperationKey = tuple<QMDDEdge, OperationType, QMDDEdge>;

using OperationResult = pair<complex<double>, size_t>;


using QMDDVariant = variant<QMDDGate, QMDDState>;
ostream& operator<<(ostream& os, const QMDDVariant& variant);


struct QMDDEdge{
    complex<double> weight;
    size_t uniqueTableKey;
    bool isTerminal;

    QMDDEdge(complex<double> w = {0.0, 0.0}, shared_ptr<QMDDNode> n = nullptr);
    QMDDEdge(double w, shared_ptr<QMDDNode> n = nullptr);
    QMDDEdge(complex<double> w, size_t key);
    QMDDEdge(double w, size_t key);
    QMDDEdge(const QMDDEdge& other) = default;
    shared_ptr<QMDDNode> getStartNode() const;
    vector<complex<double>> getAllElementsForKet();
    ~QMDDEdge() = default;
    QMDDEdge& operator=(const QMDDEdge& other) = default;
    bool operator==(const QMDDEdge& other) const;
    bool operator!=(const QMDDEdge& other) const;
    friend ostream& operator<<(ostream& os, const QMDDEdge& edge);
    QMDDEdge operator+(const QMDDEdge& other) const;
    QMDDEdge& operator+=(const QMDDEdge& other);
};

struct QMDDNode {
    vector<vector<QMDDEdge>> edges;

    QMDDNode(const vector<vector<QMDDEdge>>& edges);
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

class QMDDBase{
protected:
    QMDDEdge initialEdge;
public:
    QMDDBase(QMDDEdge edge);
    virtual ~QMDDBase() = default;
    QMDDBase(const QMDDBase& other) = default;
    QMDDBase& operator=(const QMDDBase& other) = default;
    bool operator==(const QMDDBase& other) const;
    bool operator!=(const QMDDBase& other) const;
    shared_ptr<QMDDNode> getStartNode() const;
    QMDDEdge getInitialEdge() const;
};

class QMDDGate : public QMDDBase{
public:
    QMDDGate(QMDDEdge edge);
    friend ostream& operator<<(ostream& os, const QMDDGate& gate);
};

class QMDDState : public QMDDBase{
public:
    QMDDState(QMDDEdge edge);
    vector<complex<double>> getAllElements();
    friend ostream& operator<<(ostream& os, const QMDDState& state);
};

template<typename T>
bool compare_and_swap(T& variable, const T& expected, const T& new_value) {
    if (variable == expected) {
        variable = new_value;
        return true;  // 成功
    }
    return false;  // 失敗
}

#endif
