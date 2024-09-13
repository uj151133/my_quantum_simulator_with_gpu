#include "state.cuh"

__global__ void createKet0Node(cuDoubleComplex* weights, QMDDNode* nodes) {
    weights[0] = 1.0;
    nodes[0]->edges[0][0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1][0] = QMDDEdge(.0, nullptr);
}

__global__ void createKet1Node(cuDoubleComplex* weights, QMDDNode* nodes) {
    weights[0] = 1.0;
    nodes[0]->edges[0][0] = QMDDEdge(.0, nullptr);
    nodes[0]->edges[1][0] = QMDDEdge(1.0, nullptr);
}

__global__ void createKetPlusNode(cuDoubleComplex* weights, QMDDNode* nodes) {
    weights[0] = 1.0 / sqrt(2.0);
    nodes[0]->edges[0][0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1][0] = QMDDEdge(1.0, nullptr);
}

__global__ void createKetMinusNode(cuDoubleComplex* weights, QMDDNode* nodes) {
    weights[0] = 1.0 / sqrt(2.0);
    nodes[0]->edges[0][0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[1][0] = QMDDEdge(-1.0, nullptr);
}

__global__ void createBra0Node(cuDoubleComplex* weights, QMDDNode* nodes) {
    weights[0] = 1.0;
    nodes[0]->edges[0][0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[0][1] = QMDDEdge(.0, nullptr);
}

__global__ void createBra0Node(cuDoubleComplex* weights, QMDDNode* nodes) {
    weights[0] = 1.0;
    nodes[0]->edges[0][0] = QMDDEdge(.0, nullptr);
    nodes[0]->edges[0][1] = QMDDEdge(1.0, nullptr);
}

__global__ void createBraPlusNode(cuDoubleComplex* weights, QMDDNode* nodes) {
    weights[0] = 1.0 / sqrt(2.0);
    nodes[0]->edges[0][0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[0][1] = QMDDEdge(1.0, nullptr);
}

__global__ void createBraMinusNode(cuDoubleComplex* weights, QMDDNode* nodes) {
    weights[0] = 1.0 / sqrt(2.0);
    nodes[0]->edges[0][0] = QMDDEdge(1.0, nullptr);
    nodes[0]->edges[0][1] = QMDDEdge(-1.0, nullptr);
}
/////////////////////////////////////
//
//	KET VECTORS
//
/////////////////////////////////////

QMDDState state::KetO() {
    cuDoubleComplex* weights;
    QMDDNode* nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createKet0Node<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge ket0Edge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDState(ket0Edge);
}

QMDDState state::Ket1() {
    cuDoubleComplex* weights;
    QMDDNode* nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createKet1Node<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge ket1Edge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDState(ket1Edge);
}

QMDDState state::KetPlus() {
    cuDoubleComplex* weights;
    QMDDNode* nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createKetPlusNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge ketPlusEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDState(ketPlusEdge);
}

QMDDState state::KetMinus() {
    cuDoubleComplex* weights;
    QMDDNode* nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createKetMinusNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge ketMinusEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDState(ketMinusEdge);
}

QMDDState state::Bra0() {
    cuDoubleComplex* weights;
    QMDDNode* nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createBra0Node<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge bra0Edge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDState(bra0Edge);
}

QMDDState state::Bra1() {
    cuDoubleComplex* weights;
    QMDDNode* nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createBra1Node<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge bra1Edge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDState(bra1Edge);
}

QMDDState state::BraPlus() {
    cuDoubleComplex* weights;
    QMDDNode* nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createBraPlusNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge braPlusEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDState(braPlusEdge);
}

QMDDState state::BraMinus() {
    cuDoubleComplex* weights;
    QMDDNode* nodes;
    cudaMallocManaged(&weights, sizeof(cuDoubleComplex) * 1);
    cudaMallocManaged(&nodes, sizeof(QMDDNode*) * 1);
    cudaMallocManaged(&nodes[0], sizeof(QMDDNode));
    
    createBraPlusNode<<<1, 1>>>(weights, nodes);
    cudaDeviceSynchronize();
    
    QMDDEdge braMinusEdge(weights[0], nodes[0]);

    cudaFree(weights);
    cudaFree(nodes[0]);
    cudaFree(nodes);

    return QMDDState(braMinusEdge);
}