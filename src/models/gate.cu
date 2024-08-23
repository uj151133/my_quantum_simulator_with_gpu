#include "gate.cuh"


__device__ cuDoubleComplex i = make_cuDoubleComplex(0.0, 1.0);

__global__ void createZeroNode(QMDDNode* node) {
    weights[0] = 0.0;
    nodes[0]->edges[0] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(0.0, nullptr);
}

__global__ void createIdentityNode(cuDoubleComplex* weights, cuDoubleComplex** nodes) {
    weights[0] = 1.0;
    nodes[0]->edges[0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(1.0, nullptr);
}

__global__ void createGlobalPhaseNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double delta) {
    weights[0] = cuCexp(cuCmul(i, make_cuDoubleComplex(delta, 0.0)));
    nodes[0]->edges[0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(1.0, nullptr);
}

__global__ void createPauliXNode(cuDoubleComplex* weights, cuDoubleComplex** nodes) {
    weighs[0] = 1.0
    nodes[0]->edges[0] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(0.0, nullptr);
}

__global__ void createPauliYNode(cuDoubleComplex* weights, cuDoubleComplex** nodes) {
    weighs[0] = i;
    nodes[0]->edges[0] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(0.0, nullptr);
}

__global__ void createPauliZNode(cuDoubleComplex* weights, cuDoubleComplex** nodes) {
    weighs[0] = 1.0;
    nodes[0]->edges[0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(-1.0, nullptr);
}

__global__ void createPhaseSNode(cuDoubleComplex* weights, cuDoubleComplex** nodes) {
    weighs[0] = 1.0;
    nodes[0]->edges[0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(i, nullptr);
}

__global__ void createSquareRootOfXNode(cuDoubleComplex* weights, cuDoubleComplex** nodes) {
    weighs[0] = 1.0 / 2.0 + i / 2.0;
    nodes[0]->edges[0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(i, nullptr);
    nodes[0]->edges[2] = QMDDEdge(i, nullptr);
    nodes[0]->edges[3] = QMDDEdge(1.0, nullptr);
}

__global__ void createHadamardNode(cuDoubleComplex* weights, cuDoubleComplex** nodes) {
    weighs[0] = 1.0 / sqrt(2.0);
    nodes[0]->edges[0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(-1.0, nullptr);
}

__global__ void createPhaseShiftNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double phi) {
    weighs[0] = 1.0;
    nodes[0]->edges[0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(cuCexp(cuCmul(i, make_cuDoubleComplex(phi, 0.0))), nullptr);
}

__global__ void createPhaseTNode(cuDoubleComplex* weights, cuDoubleComplex** nodes) {
    weighs[0] = 1.0;
    nodes[0]->edges[0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(cuCexp(cuCmul(i, make_cuDoubleComplex(M_PI / 4.0, 0.0))), nullptr);
}

__global__ void createRotationAboutXNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta) {
    weighs[0] = 1.0;
    nodes[0]->edges[0] = QMDDEdge(cos(theta / 2.0), nullptr);
    nodes[0]->edges[1] = QMDDEdge(-i * sin(theta / 2.0), nullptr);
    nodes[0]->edges[2] = QMDDEdge(-i * sin(theta / 2.0), nullptr);
    nodes[0]->edges[3] = QMDDEdge(cos(theta / 2.0), nullptr);
}

__global__ void create RotationAboutYNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta) {
    weighs[0] = 1.0;
    nodes[0]->edges[0] = QMDDEdge(cos(theta / 2.0), nullptr);
    nodes[0]->edges[1] = QMDDEdge(-sin(theta / 2.0), nullptr);
    nodes[0]->edges[2] = QMDDEdge(sin(theta / 2.0), nullptr);
    nodes[0]->edges[3] = QMDDEdge(cos(theta / 2.0), nullptr);
}

__global__ void createRotationAboutZNode(cuDoubleComplex* weights, cuDoubleComplex** nodes, double theta) {
    weighs[0] = 1.0;
    nodes[0]->edges[0] = QMDDEdge(cuCexp(cuCmul(-i, make_cuDoubleComplex(theta / 2.0, 0.0))), nullptr);
    nodes[0]->edges[1] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[2] = QMDDEdge(0.0, nullptr);
    nodes[0]->edges[3] = QMDDEdge(cuCexp(cuCmul(i, make_cuDoubleComplex(theta / 2.0, 0.0))), nullptr);
}

QMDDGate gate::ZERO() {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createZeroNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge zeroEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(zeroEdge);
}

QMDDGate gate::I() {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createIdentityNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge iEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(iEdge);
}


QMDDGate gate::Ph(double delta) {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createGlobalPhaseNode<<<1, 1>>>(weights, nodes, delta);
    cudaDeviceSynchronize();
    
    QMDDEdge phEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(phEdge);
}

QMDDGate gate::X() {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createPauliXNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge xEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(xEdge);
}

QMDDGate gate::Y() {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createPauliYNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge yEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(yEdge);
}

QMDDGate gate::Z() {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createPauliZNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge zEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(zEdge);
}

QMDDGate gate::S() {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createPhaseSNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge sEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(sEdge);
}

QMDDGate gate::V() {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createSquareRootOfXNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge vEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(vEdge);
}

QMDDGate gate::H() {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createHadamardNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge hEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(hEdge);
}

QMDDGate gate::P(double phi) {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createPhaseShiftNode<<<1, 1>>>(weights, nodes, phi);
    cudaDeviceSynchronize();
    
    QMDDEdge pEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(pEdge);
}

QMDDGate gate::T() {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createPhaseTNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge tEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(tEdge);
}

QMDDGate gate::Rx(double theta) {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createRotationAboutXNode<<<1, 1>>>(weights, nodes, theta);
    cudaDeviceSynchronize();
    
    QMDDEdge rxEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(rxEdge);
}

QMDDGate gate::Ry(double theta) {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createRotationAboutYNode<<<1, 1>>>(weights, nodes, theta);
    cudaDeviceSynchronize();
    
    QMDDEdge rxEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(rxEdge);
}

QMDDGate gate::Rz(double theta) {
    cuDoubleComplex* weights;
    cuDoubleComplex** nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createRotationAboutZNode<<<1, 1>>>(weights, nodes, theta);
    cudaDeviceSynchronize();
    
    QMDDEdge rzEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDGate(rzEdge);
}

QMDDGate Ph(double delta) {
    cuDoubleComplex* weights;
    QMDDNode** nodes;
    
    // メモリの確保
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));

    // カーネルの呼び出し
    createPhaseNode<<<1, 1>>>(weights, nodes, delta);
    cudaDeviceSynchronize();

    // QMDDEdge の作成
    QMDDEdge phEdge(weights[0], nodes[0]);
    
    // メモリの解放
    cudaFree(nodes[0]);  // ノードのメモリ解放
    cudaFree(nodes);     // ノード配列のメモリ解放
    cudaFree(weights);   // ウェイトのメモリ解放

    // QMDDGate の作成
    return QMDDGate(phEdge);
}
