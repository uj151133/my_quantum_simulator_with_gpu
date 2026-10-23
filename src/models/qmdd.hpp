#ifndef QMDD_HPP
#define QMDD_HPP

#include <atomic>
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
#include "arrayLeaf.hpp"

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
    ArrayLeaf
};

// using OperationKey = tuple<QMDDEdge, OperationType, QMDDEdge>;
using OperationKey = pair<int64_t, int64_t>;

using OperationResult = pair<complex<double>, int64_t>;

struct QMDDEdge{
    complex<double> weight = complex<double>(.0, .0);
    int64_t key_ = 0;
    uintptr_t son_ = 0;
    SonKind sonKind_ = SonKind::Terminal;
    bool isTerminal = true;

    QMDDEdge();
    QMDDEdge(complex<double> w);
    QMDDEdge(double w);
    QMDDEdge(complex<double> w, QMDDNode* node);
    QMDDEdge(double w, QMDDNode* node);
    QMDDEdge(complex<double> w, ArrayLeaf* leaf);
    QMDDEdge(double w, ArrayLeaf* leaf);
    QMDDEdge(complex<double> w, int64_t key, QMDDNode* node);
    QMDDEdge(double w, int64_t key, QMDDNode* node);
    QMDDEdge(complex<double> w, int64_t key, ArrayLeaf* leaf);
    QMDDEdge(double w, int64_t key, ArrayLeaf* leaf);
    QMDDEdge(complex<double> w, int64_t key, SonKind kind);
    QMDDEdge(double w, int64_t key, SonKind kind);
    QMDDEdge(const QMDDEdge& other)
        : weight(other.weight), key_(other.key_), son_(other.son_), sonKind_(other.sonKind_), isTerminal(other.isTerminal) { this->retain_(); }
    QMDDEdge& operator=(const QMDDEdge& other) {
        if (this != &other) {
            other.retain_();
            this->release_();
            this->weight = other.weight; this->key_ = other.key_; this->son_ = other.son_; this->sonKind_ = other.sonKind_; this->isTerminal = other.isTerminal;
        }
        return *this;
    }
    // ムーブは所有を移すだけで参照カウントを触らない（コピーはアトミックな +1、元が消えるとき −1 が要る）。
    // 移した後の元は「重み 0 の終端」にしておくので、そのデストラクタは何もしない。
    // noexcept にしておかないと vector の再確保でムーブが使われない。
    QMDDEdge(QMDDEdge&& other) noexcept
        : weight(other.weight), key_(other.key_), son_(other.son_), sonKind_(other.sonKind_), isTerminal(other.isTerminal) {
        other.weight = complex<double>(.0, .0); other.key_ = 0; other.son_ = 0;
        other.sonKind_ = SonKind::Terminal; other.isTerminal = true;
    }
    QMDDEdge& operator=(QMDDEdge&& other) noexcept {
        if (this != &other) {
            this->release_();
            this->weight = other.weight; this->key_ = other.key_; this->son_ = other.son_; this->sonKind_ = other.sonKind_; this->isTerminal = other.isTerminal;
            other.weight = complex<double>(.0, .0); other.key_ = 0; other.son_ = 0;
            other.sonKind_ = SonKind::Terminal; other.isTerminal = true;
        }
        return *this;
    }
    ~QMDDEdge() { this->release_(); }

    QMDDNode* node() const { return this->sonKind_ == SonKind::QMDDNode ? reinterpret_cast<QMDDNode*>(this->son_ & ~uintptr_t(3)) : nullptr; }
    ArrayLeaf* leaf() const { return this->sonKind_ == SonKind::ArrayLeaf   ? reinterpret_cast<ArrayLeaf*>(this->son_ & ~uintptr_t(3))   : nullptr; }

    vector<complex<double>> openKet();
    bool operator==(const QMDDEdge& other) const;
    bool operator!=(const QMDDEdge& other) const;
    friend ostream& operator<<(ostream& os, const QMDDEdge& edge);
    void Noah();
    int level() const;
private:
    void retain_() const;    // 子の参照カウント +1（終端なら何もしない）
    void release_();         // 子の参照カウント -1。0 になったら解放
};


using EdgeMatrix = std::array<std::array<QMDDEdge, 2>, 2>;

struct QMDDNode {
    EdgeMatrix edges;
    int level = 0;   // このノードが担当する量子ビットの位置（1 始まり）。= 1 + 子の level の最大。生成時に確定し、以後不変
    int64_t key = 0;                      // ユニークテーブルに登録されたときのキー（0 = 未登録）
    std::atomic<uint32_t> refCount{0};    // このノードを指すエッジの数。0 になった瞬間にテーブルから外して解放
    QMDDNode* next = nullptr;             // ユニークテーブルのバケット内の鎖

    explicit QMDDNode(const EdgeMatrix& edges);
    explicit QMDDNode(EdgeMatrix&& edges) noexcept;
    QMDDNode(const vector<vector<QMDDEdge>>& edges);
    ~QMDDNode() = default;
    QMDDNode(const QMDDNode& other) : edges(other.edges), level(other.level) {}   // 内容だけ複製（登録情報は写さない）
    QMDDNode& operator=(const QMDDNode& other) = delete;
    QMDDNode(QMDDNode&& other) = delete;
    QMDDNode& operator=(QMDDNode&& other) = delete;
    bool operator==(const QMDDNode& other) const;
    bool operator!=(const QMDDNode& other) const;
    friend ostream& operator<<(ostream& os, const QMDDNode& node);
    vector<complex<double>> getWeights() const;
};
void disableEdgeReleaseAtExit();
template <class... Args>
inline QMDDNode* bet(Args&&... args) { return new QMDDNode(std::forward<Args>(args)...); }
ArrayLeaf* pluck(size_t dim, void* reBuf, void* imBuf, int64_t id);

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
