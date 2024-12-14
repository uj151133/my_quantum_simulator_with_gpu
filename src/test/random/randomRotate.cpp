#include "randomRotate.hpp"

void randomRotate(size_t numQubits, size_t numGates = 200) {

    QuantumCircuit circuit(numQubits);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> qubitDist(0, numQubits - 1);
    uniform_real_distribution<double> phaseDist(0.0, 2.0 * M_PI);
    uniform_int_distribution<int> gateDist(0, 2);

    for (int i = 0; i < numGates; ++i) {
        int qubit = qubitDist(gen);
        double phase = phaseDist(gen);
        int gateType = gateDist(gen);
        cout <<"gate num: " << i << ", qubit: " << qubit << ", phase: " << phase << ", gateType: " << gateType << endl;

        switch (gateType) {
            case 0:
                circuit.addRx(qubit, phase);
                break;
            case 1:
                circuit.addRy(qubit, phase);
                break;
            case 2:
                circuit.addRz(qubit, phase);
                break;
        }
    }

    circuit.execute();
}