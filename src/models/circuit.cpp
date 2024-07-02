#include <iostream>
#include <complex>
#include <array>
#include <unordered_map>
#include <functional>

// QuantumGateクラスの定義
class QuantumGate {
private:
    QMDDNode* node;
    QMDDEdge initialEdge;

public:
    QuantumGate(QMDDNode* n, std::complex<double> weight)
        : node(n), initialEdge(weight, n) {}

    ~QuantumGate() {
        delete node;
    }

    QMDDNode* getNode() const {
        return node;
    }

    QMDDEdge getInitialEdge() const {
        return initialEdge;
    }
};

// QuantumCircuitクラスの定義
class QuantumCircuit {
private:
    std::vector<QuantumGate*> gates;

public:
    ~QuantumCircuit() {
        for (auto gate : gates) {
            delete gate;
        }
    }

    void addGate(QuantumGate* gate) {
        gates.push_back(gate);
    }

    void printCircuit() const {
        for (size_t i = 0; i < gates.size(); ++i) {
            std::cout << "Gate " << i << ":\n";
            auto node = gates[i]->getNode();
            auto initialEdge = gates[i]->getInitialEdge();
            std::cout << "Initial Edge Weight: " << initialEdge.weight << "\n";
            for (size_t j = 0; j < node->edges.size(); ++j) {
                std::cout << "Edge " << j << ": Weight: " << node->edges[j].weight
                          << ", Node: " << node->edges[j].node << "\n";
            }
        }
    }
};


