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
#include <boost/fiber/all.hpp>
#include <stack>
#include "sv.hpp"
#include "memo.hpp"

using namespace std;

struct QMDDNode;

struct QMDDEdge;

class QMDDGate;

class QMDDState;


enum class OperationType {
    ADD,
    MUL,
    KRONECKER,
    TEST,
};

enum class SonKind : uint8_t {
    QMDDNode,
    SVLeaf,
    Terminal
};

// using OperationKey = tuple<QMDDEdge, OperationType, QMDDEdge>;
using OperationKey = pair<int64_t, int64_t>;

using OperationResult = pair<complex<double>, int64_t>;

using QMDDVariant = variant<QMDDGate, QMDDState>;
ostream& operator<<(ostream& os, const QMDDVariant& variant);


struct QMDDEdge{
    complex<double> weight = complex<double>(.0, .0);
    int64_t key_ = 0;
    shared_ptr<QMDDNode> sonNode_ = nullptr;
    shared_ptr<SVLeaf> sonLeaf_ = nullptr;
    SonKind sonKind_ = SonKind::Terminal;
    bool isTerminal = true;
    int depth = 0;

    QMDDEdge();
    QMDDEdge(complex<double> w);
    QMDDEdge(double w);
    QMDDEdge(complex<double> w, shared_ptr<QMDDNode> n);
    QMDDEdge(double w, shared_ptr<QMDDNode> n);
    QMDDEdge(complex<double> w, int64_t key, shared_ptr<SVLeaf> l);
    QMDDEdge(double w, int64_t key, shared_ptr<SVLeaf> l);
    QMDDEdge(complex<double> w, int64_t key, SonKind kind);
    QMDDEdge(double w, int64_t key, SonKind kind);
    QMDDEdge(const QMDDEdge& other) = default;
    shared_ptr<QMDDNode> getStartNode() const;
    shared_ptr<SVLeaf> getStartLeaf() const;
    vector<complex<double>> getAllElementsForKet();
    ~QMDDEdge() = default;
    QMDDEdge& operator=(const QMDDEdge& other) = default;
    bool operator==(const QMDDEdge& other) const;
    bool operator!=(const QMDDEdge& other) const;
    friend ostream& operator<<(ostream& os, const QMDDEdge& edge);
    void calculateDepth();
};

struct QMDDNode {
    vector<vector<QMDDEdge>> edges;

    QMDDNode(const vector<vector<QMDDEdge>>& edges);
    ~QMDDNode() = default;
    QMDDNode(const QMDDNode& other) = default;
    QMDDNode& operator=(const QMDDNode& other) = default;
    QMDDNode(QMDDNode&& other) noexcept = default;
    QMDDNode& operator=(QMDDNode&& other) noexcept;
    bool operator==(const QMDDNode& other) const;
    bool operator!=(const QMDDNode& other) const;
    friend ostream& operator<<(ostream& os, const QMDDNode& node);
    vector<complex<double>> getWeights() const;
};

class QMDDGate{
private:
    QMDDEdge initialEdge_;
public:
    QMDDGate(QMDDEdge edge = QMDDEdge());
    QMDDGate(const QMDDGate& other) = default;
    ~QMDDGate() = default;
    // shared_ptr<QMDDNode> getStartNode() const;
    QMDDEdge getInitialEdge() const;
    QMDDGate& operator=(const QMDDGate& other) = default;
    bool operator==(const QMDDGate& other) const;
    bool operator!=(const QMDDGate& other) const;
    friend ostream& operator<<(ostream& os, const QMDDGate& gate);
};

class QMDDState{
private:
    QMDDEdge initialEdge_;
public:
    QMDDState(QMDDEdge edge);
    QMDDState(const QMDDState& other) = default;
    ~QMDDState() = default;
    // shared_ptr<QMDDNode> getStartNode() const;
    QMDDEdge getInitialEdge() const;
    vector<complex<double>> getAllElements();
    QMDDState& operator=(const QMDDState& other) = default;
    bool operator==(const QMDDState& other) const;
    bool operator!=(const QMDDState& other) const;
    friend ostream& operator<<(ostream& os, const QMDDState& state);
};

class QMDDSuite{
    private:
    QMDDEdge initialEdge_;
public:
    QMDDSuite(QMDDEdge edge = QMDDEdge());
    QMDDSuite(const QMDDSuite& other) = default;
    ~QMDDSuite() = default;
    QMDDEdge getInitialEdge() const;
    vector<complex<double>> getAllElements();
    QMDDSuite& operator=(const QMDDSuite& other) = default;
    bool operator==(const QMDDSuite& other) const;
    bool operator!=(const QMDDSuite& other) const;
    friend ostream& operator<<(ostream& os, const QMDDSuite& suite);
};

#endif
