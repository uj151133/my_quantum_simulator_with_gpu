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

using OperationKey = tuple<QMDDEdge, OperationType, QMDDEdge>;

using OperationResult = pair<complex<double>, int64_t>;

using QMDDVariant = variant<QMDDGate, QMDDState>;
ostream& operator<<(ostream& os, const QMDDVariant& variant);

extern "C" {
    struct CComplex {
        double real;
        double imag;
    };

    struct CQMDDEdge {
        CComplex weight;
        uint64_t uniqueTableKey;
        bool isTerminal;
        int depth;
    };
    struct CQMDDNode {
        CQMDDEdge* edges_ptr;
        uint rows;
        uint cols;
    };

    const CQMDDNode* find(uint64_t unique_id);
    bool insert(uint64_t unique_id, const CQMDDNode* node);

    void init_unique_table();
    void shutdown_unique_table();
}

struct QMDDEdge{
    complex<double> weight;
    int64_t uniqueTableKey;
    bool isTerminal;
    int depth;

    QMDDEdge(complex<double> w = {.0, .0}, shared_ptr<QMDDNode> n = nullptr);
    QMDDEdge(double w, shared_ptr<QMDDNode> n = nullptr);
    QMDDEdge(complex<double> w, int64_t key);
    QMDDEdge(double w, int64_t key);
    QMDDEdge(const QMDDEdge& other) = default;
    shared_ptr<QMDDNode> getStartNode() const;
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
    QMDDEdge initialEdge;
public:
    QMDDGate(QMDDEdge edge);
    QMDDGate(const QMDDGate& other) = default;
    ~QMDDGate() = default;
    shared_ptr<QMDDNode> getStartNode() const;
    QMDDEdge getInitialEdge() const;
    QMDDGate& operator=(const QMDDGate& other) = default;
    bool operator==(const QMDDGate& other) const;
    bool operator!=(const QMDDGate& other) const;
    friend ostream& operator<<(ostream& os, const QMDDGate& gate);
};

class QMDDState{
private:
    QMDDEdge initialEdge;
public:
    QMDDState(QMDDEdge edge);
    QMDDState(const QMDDState& other) = default;
    ~QMDDState() = default;
    shared_ptr<QMDDNode> getStartNode() const;
    QMDDEdge getInitialEdge() const;
    vector<complex<double>> getAllElements();
    QMDDState& operator=(const QMDDState& other) = default;
    bool operator==(const QMDDState& other) const;
    bool operator!=(const QMDDState& other) const;
    friend ostream& operator<<(ostream& os, const QMDDState& state);
};

#endif
