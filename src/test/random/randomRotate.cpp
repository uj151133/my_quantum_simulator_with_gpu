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

    circuit.simulate();
}

void randomRotateDeep(size_t numQubits, size_t numGates = 200) {

    QuantumCircuit circuit(numQubits);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> phaseDist(0.0, 2.0 * M_PI);
    uniform_int_distribution<int> gateDist(0, 1);

    for (int i = 0; i < numGates; ++i) {
        double phase = phaseDist(gen);
        int gateType = gateDist(gen);

        switch (gateType) {
            case 0:
                circuit.addRx(numQubits - 1, phase);
                break;
            case 1:
                circuit.addRy(numQubits - 1, phase);
                break;
        }
    }

    circuit.simulate();
}

void random2(size_t numQubits, size_t numGates) {
    QuantumCircuit circuit(numQubits);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> qubitDist(0, numQubits - 1);
    uniform_real_distribution<double> phaseDist(0.0, 2.0 * M_PI);
    uniform_int_distribution<int> gateDist(0, 7);

    for (size_t i = 0; i < numGates; ++i) {
        int gateType = gateDist(gen);
        double phase = phaseDist(gen);
        int qubit = qubitDist(gen);

        switch (gateType) {
            case 0:
                circuit.addPh(qubit, phase);
                break;
            case 1:
                circuit.addX(qubit);
                break;
            case 2:
                circuit.addY(qubit);
                break;
            case 3:
                circuit.addZ(qubit);
                break;
            case 4:
                circuit.addS(qubit);
                break;
            case 5:
                circuit.addP(qubit, phase);
                break;
            case 6:
                circuit.addT(qubit);
                break;
            case 7:
                circuit.addRz(qubit, phase);
                break;
        }
    }

    circuit.simulate();
}

void random4(size_t numQubits, size_t numGates) {
    QuantumCircuit circuit(numQubits);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> qubitDist(0, numQubits - 1);
    uniform_real_distribution<double> phaseDist(0.0, 2.0 * M_PI);
    uniform_int_distribution<int> gateDist(0, 3);

    for (size_t i = 0; i < numGates; ++i) {
        int gateType = gateDist(gen);
        double phase = phaseDist(gen);
        int qubit = qubitDist(gen);

        if (numQubits == 0) continue;

        switch (gateType) {
            case 0:
                circuit.addH(qubit);
                break;
            case 1:
                circuit.addV(qubit);
                break;
            case 2:
                circuit.addRx(qubit, phase);
                break;
            case 3:
                circuit.addRy(qubit, phase);
                break;
        }
    }

    circuit.simulate();
}