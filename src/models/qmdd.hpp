#ifndef QMDD_HPP
#define QMDD_HPP

#include <variant>
#include <iostream>
#include <complex>
#include <cstring>
#include <array>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <memory>
#include <tuple>
#include <boost/fiber/all.hpp>
#include <stack>
#include <limits> 
#include "sv.hpp"

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
    Terminal,
    QMDDNode,
    SVLeaf
};

// using OperationKey = tuple<QMDDEdge, OperationType, QMDDEdge>;
using OperationKey = pair<int64_t, int64_t>;

using OperationResult = pair<complex<double>, int64_t>;

using SonVariant = variant<monostate, shared_ptr<QMDDNode>, shared_ptr<SVLeaf>>;


struct QMDDEdge{
    // complex<double> weight = complex<double>(.0, .0);
    double magnitude = -std::numeric_limits<double>::infinity();
    double angle = std::numeric_limits<double>::quiet_NaN();
    int64_t key_ = 0;
    SonVariant son_ = monostate();
    SonKind sonKind_ = SonKind::Terminal;
    bool isTerminal = true;
    int depth = 0;

    QMDDEdge();
    QMDDEdge(const pair<double, double>& polar);
    QMDDEdge(const pair<double, double>& polar, SonVariant son);
    QMDDEdge(const pair<double, double>& polar, int64_t key, SonVariant son);
    QMDDEdge(const pair<double, double>& polar, int64_t key, SonKind kind);
    QMDDEdge(const QMDDEdge& other) = default;
    SonVariant getSon() const;
    vector<complex<double>> openKet();
    ~QMDDEdge() = default;
    QMDDEdge& operator=(const QMDDEdge& other) = default;
    bool operator==(const QMDDEdge& other) const;
    bool operator!=(const QMDDEdge& other) const;
    friend ostream& operator<<(ostream& os, const QMDDEdge& edge);
    void Noah();
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
    QMDDSuite(const pair<double, double>& polar, SonVariant son);
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
