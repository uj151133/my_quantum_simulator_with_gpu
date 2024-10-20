#include "gate.hpp"

static complex<double> i(0.0, 1.0);

QMDDGate gate::O() {
    // return QMDDGate(QMDDEdge(.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
    //     {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
    //     {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
    // })));
    return QMDDGate(QMDDEdge(.0, nullptr));
}

QMDDGate gate::I() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    })));
}

QMDDGate gate::Ph(double delta) {
    return QMDDGate(QMDDEdge(exp(i * delta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
    })));
}

QMDDGate gate::X() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
    })));
}

// const QMDDGate gate::PLUS_X_GATE = [] {
//     complex<double> plusXWeight = 1 / sqrt(2.0);
//     shared_ptr<QMDDNode> plusXNode = make_shared<QMDDNode>(4);

//     plusXNode->children[0] = QMDDEdge(1, nullptr);
//     plusXNode->children[1] = QMDDEdge(i, nullptr);
//     plusXNode->children[2] = QMDDEdge(i, nullptr);
//     plusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusXEdge(plusXWeight, plusXNode);
//     return QMDDGate(plusXEdge);
// }();

// const QMDDGate gate::MINUS_X_GATE = [] {
//     complex<double> minusXWeight = 1 / sqrt(2.0);
//     shared_ptr<QMDDNode> minusXNode = make_shared<QMDDNode>(4);

//     minusXNode->children[0] = QMDDEdge(1, nullptr);
//     minusXNode->children[1] = QMDDEdge(-i, nullptr);
//     minusXNode->children[2] = QMDDEdge(-i, nullptr);
//     minusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusXEdge(minusXWeight, minusXNode);
//     return QMDDGate(minusXEdge);
// }();

QMDDGate gate::Y() {
    return QMDDGate(QMDDEdge(-i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(-1.0, nullptr), QMDDEdge(.0, nullptr)}
    })));
}

// QMDDGate createPlusYGate() {
//     complex<double> plusYWeight = 1 / sqrt(2.0);
//     QMDDNode* plusYNode = new QMDDNode(4);

//     plusYNode->children[0] = QMDDEdge(1, nullptr);
//     plusYNode->children[1] = QMDDEdge(1, nullptr);
//     plusYNode->children[2] = QMDDEdge(-1, nullptr);
//     plusYNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusYEdge(plusYWeight, plusYNode);
//     return QMDDGate(plusYEdge);
// }

// QMDDGate createMinusYGate() {
//     complex<double> minusYWeight = 1 / sqrt(2.0);
//     QMDDNode* minusYNode = new QMDDNode(4);

//     minusYNode->children[0] = QMDDEdge(1, nullptr);
//     minusYNode->children[1] = QMDDEdge(-1, nullptr);
//     minusYNode->children[2] = QMDDEdge(1, nullptr);
//     minusYNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusYEdge(minusYWeight, minusYNode);
//     return QMDDGate(minusYEdge);
// }
QMDDGate gate::Z() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(-1.0, nullptr)}
    })));
}

QMDDGate gate::S() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(i, nullptr)}
    })));
}

QMDDGate gate::Sdagger() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(-i, nullptr)}
    })));
}

QMDDGate gate::V() {
    return QMDDGate(QMDDEdge(1.0 / 2.0 + i / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
        {QMDDEdge(i, nullptr), QMDDEdge(1.0, nullptr)}
    })));
}

QMDDGate gate::H() {
    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(1.0, nullptr)},
        {QMDDEdge(1.0, nullptr), QMDDEdge(-1.0, nullptr)}
    })));
}

QMDDGate gate::CX1() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))}
    })));
}

QMDDGate gate::CX2() {
    shared_ptr<QMDDNode> cx2Node1, cx2Node2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            cx2Node1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            cx2Node2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, cx2Node1), QMDDEdge(1.0, cx2Node2)},
        {QMDDEdge(1.0, cx2Node2), QMDDEdge(1.0, cx2Node1)}
    })));
}

QMDDGate gate::varCX() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    })));
}

QMDDGate gate::CZ() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::Z().getStartNode()))}
    })));
}

QMDDGate gate::DCNOT() {
    shared_ptr<QMDDNode> dcnotNode1, dcnotNode2, dcnotNode3, dcnotNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            dcnotNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            dcnotNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            dcnotNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
            });
        }

        #pragma omp section
        {
            dcnotNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dcnotNode1), QMDDEdge(1.0, dcnotNode2)},
        {QMDDEdge(1.0, dcnotNode3), QMDDEdge(1.0, dcnotNode4)}
    })));
}

QMDDGate gate::SWAP(bool primitive) {
    if (primitive) {
        return QMDDGate(mathUtils::mul(mathUtils::mul(gate::CX1().getInitialEdge(), gate::CX2().getInitialEdge()), gate::CX1().getInitialEdge()));
    } else {
        shared_ptr<QMDDNode> swapNode1, swapNode2, swapNode3, swapNode4;
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                swapNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                    {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                    {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
                });
            }

            #pragma omp section
            {
                swapNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                    {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                    {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
                });
            }

            #pragma omp section
            {
                swapNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                    {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                    {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
                });
            }

            #pragma omp section
            {
                swapNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                    {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                    {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
                });
            }
        }

        #pragma omp barrier

        return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, swapNode1), QMDDEdge(1.0, swapNode2)},
            {QMDDEdge(1.0, swapNode3), QMDDEdge(1.0, swapNode4)}
        })));
    }
}

QMDDGate gate::iSWAP() {
    shared_ptr<QMDDNode> iswapNode1, iswapNode2, iswapNode3, iswapNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            iswapNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            iswapNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            iswapNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            iswapNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, iswapNode1), QMDDEdge(i, iswapNode2)},
        {QMDDEdge(i, iswapNode3), QMDDEdge(1.0, iswapNode4)}
    })));
}

QMDDGate gate::P(double phi) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * phi), nullptr)}
    })));
}

QMDDGate gate::T() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * M_PI / 4.0), nullptr)}
    })));
}

QMDDGate gate::Tdagger() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(-i * M_PI / 4.0), nullptr)}
    })));
}

QMDDGate gate::CP(double phi) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode()))},
        {QMDDEdge(.0, make_shared<QMDDNode>(*gate::O().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::P(phi).getStartNode()))}
    })));
}

QMDDGate gate::CS() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::S().getStartNode()))}
    })));
}

QMDDGate gate::Rx(double theta) {
    return QMDDGate(QMDDEdge(cos(theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-i * tan(theta / 2.0), nullptr)},
        {QMDDEdge(-i * tan(theta / 2.0), nullptr), QMDDEdge(1.0, nullptr)}
    })));
}

QMDDGate gate::Ry(double theta) {
    return QMDDGate(QMDDEdge(cos(theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-tan(theta / 2.0), nullptr)},
        {QMDDEdge(tan(theta / 2.0), nullptr), QMDDEdge(1.0, nullptr)}
    })));
}

QMDDGate gate::Rz(double theta) {
    return QMDDGate(QMDDEdge(exp(-i * theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * theta), nullptr)}
    })));
}

QMDDGate gate::Rxx(double phi) {
    return QMDDGate(QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(-i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::X().getStartNode()))},
        {QMDDEdge(-i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::X().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    })));
}

QMDDGate gate::Ryy(double phi) {
    return QMDDGate(QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::Y().getStartNode()))},
        {QMDDEdge(-i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::Y().getStartNode())), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    })));
}

QMDDGate gate::Rzz(double phi) {
    return QMDDGate(QMDDEdge(exp(-i * phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::P(phi).getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * phi), make_shared<QMDDNode>(*gate::P(-phi).getStartNode()))}
    })));
}

QMDDGate gate::Rxy(double phi) {
    shared_ptr<QMDDNode> rxyNode1, rxyNode2, rxyNode3, rxyNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            rxyNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(cos(phi / 2.0), nullptr)}
            });
        }

        #pragma omp section
        {
            rxyNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            rxyNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            rxyNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0 * mathUtils::sec(phi / 2.0), nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, rxyNode1), QMDDEdge(-i * sin(phi / 2.0), rxyNode2)},
        {QMDDEdge(-i * sin(phi / 2.0), rxyNode3), QMDDEdge(cos(phi / 2.0), rxyNode4)}
    })));
}

QMDDGate gate::SquareSWAP() {
    shared_ptr<QMDDNode> squareSWAPNode1, squareSWAPNode2, squareSWAPNode3, squareSWAPNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            squareSWAPNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge((1.0 + i) / 2.0, nullptr)}
            });
        }

        #pragma omp section
        {
            squareSWAPNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            squareSWAPNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            squareSWAPNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0 - i, nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, squareSWAPNode1), QMDDEdge((1.0 - i) / 2.0, squareSWAPNode2)},
        {QMDDEdge((1.0 - i) / 2.0, squareSWAPNode3), QMDDEdge((1.0 + i) / 2.0, squareSWAPNode4)}
    })));
}

QMDDGate gate::SquareiSWAP() {
    shared_ptr<QMDDNode> squareiSWAPNode1, squareiSWAPNode2, squareiSWAPNode3, squareiSWAPNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            squareiSWAPNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0 / sqrt(2.0), nullptr)}
            });
        }

        #pragma omp section
        {
            squareiSWAPNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            squareiSWAPNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            squareiSWAPNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(sqrt(2.0), nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, squareiSWAPNode1), QMDDEdge(i / sqrt(2.0), squareiSWAPNode2)},
        {QMDDEdge(i / sqrt(2.0), squareiSWAPNode3), QMDDEdge(1.0 / sqrt(2.0), squareiSWAPNode4)}
    })));
}

QMDDGate gate::SWAPalpha(double alpha) {
    shared_ptr<QMDDNode> SWAPalphaNode1, SWAPalphaNode2, SWAPalphaNode3, SWAPalphaNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            SWAPalphaNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, nullptr)}
            });
        }

        #pragma omp section
        {
            SWAPalphaNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            SWAPalphaNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            SWAPalphaNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(2.0 / (1.0 + exp(i * M_PI * alpha)), nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, SWAPalphaNode1), QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode2)},
        {QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode3), QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, SWAPalphaNode4)}
    })));
}

QMDDGate gate::FREDKIN() {
    shared_ptr<QMDDNode> fredkinNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fredkinNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::SWAP().getStartNode()))}
    })));
}

QMDDGate gate::U(double theta, double phi, double lambda) {
    return QMDDGate(QMDDEdge(cos(theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-exp(i * lambda) * tan(theta / 2.0), nullptr)},
        {QMDDEdge(exp(i * phi) * tan(theta / 2.0), nullptr), QMDDEdge(exp(i * (lambda + phi)), nullptr)}
    })));
}

QMDDGate gate::BARENCO(double alpha, double phi, double theta) {
    shared_ptr<QMDDNode> barencoNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(-i * exp(-i * phi) * tan(theta), nullptr)},
        {QMDDEdge(-i * exp(i * phi) * tan(theta), nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * alpha) * cos(theta), barencoNode1)}
    })));
}

QMDDGate gate::B() {
    shared_ptr<QMDDNode> bNode1, bNode2, bNode3, bNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            bNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(cos(3.0 * M_PI / 8.0) * mathUtils::sec(M_PI / 8.0), nullptr)}
            });
        }

        #pragma omp section
        {
            bNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(sin(3.0 * M_PI / 8.0) * mathUtils::csc(M_PI / 8.0), nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            bNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(sin(M_PI / 8.0) * mathUtils::csc(3.0 * M_PI / 8.0), nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            bNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(cos(M_PI / 8.0) * mathUtils::sec(3.0 * M_PI / 8.0), nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(cos(M_PI / 8.0), bNode1), QMDDEdge(i * sin(M_PI / 8.0), bNode2)},
        {QMDDEdge(i * sin(3.0 * M_PI / 8.0), bNode3), QMDDEdge(cos(3.0 * M_PI / 8.0), bNode4)}
    })));
}

QMDDGate gate::CSX() {
    shared_ptr<QMDDNode> csxNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nullptr), QMDDEdge(exp(-i * M_PI / 2.0), nullptr)},
        {QMDDEdge(exp(-i * M_PI / 2.0), nullptr), QMDDEdge(1.0, nullptr)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * M_PI / 4.0), csxNode1)}
    })));
}

QMDDGate gate::N(double a, double b, double c) {
    shared_ptr<QMDDNode> nNode1, nNode2, nNode3, nNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            nNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(exp(-2.0 * i * c) * cos(a + b) * mathUtils::sec(a - b), nullptr)}
            });
        }

        #pragma omp section
        {
            nNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(exp(-2.0 * i * c) * sin(a + b) * mathUtils::csc(a - b), nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            nNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(exp(2.0 * i * c) * sin(a - b) * mathUtils::csc(a + b), nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            nNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(exp(2.0 * i * c) * cos(a - b) * mathUtils::sec(a + b), nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(exp(i * c) * cos(a - b), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nNode1), QMDDEdge(i * tan(a - b), nNode2)},
        {QMDDEdge(i * exp(-2.0 * i * c) * sin(a + b) * mathUtils::sec(a - b), nNode3), QMDDEdge(exp(-2.0 * i * c) * cos(a + b) * mathUtils::sec(a - b), nNode4)}
    })));
}

QMDDGate gate::DB() {
    shared_ptr<QMDDNode> dbNode1, dbNode2, dbNode3, dbNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            dbNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(cos(3.0 * M_PI / 8.0), nullptr)}
            });
        }

        #pragma omp section
        {
            dbNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            dbNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            dbNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0 * mathUtils::sec(3.0 * M_PI / 8.0), nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dbNode1), QMDDEdge(-i * sin(3.0 * M_PI / 8.0), dbNode2)},
        {QMDDEdge(-i * sin(3.0 * M_PI / 8.0), dbNode3), QMDDEdge(cos(3.0 * M_PI / 8.0), dbNode4)}
    })));
}

QMDDGate gate::ECR() {
    shared_ptr<QMDDNode> ecrNode1, ecrNode2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            ecrNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
                {QMDDEdge(i, nullptr), QMDDEdge(1.0, nullptr)}
            });
        }

        #pragma omp section
        {
            ecrNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)},
                {QMDDEdge(-i, nullptr), QMDDEdge(1.0, nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, ecrNode1)},
        {QMDDEdge(1.0, ecrNode2), QMDDEdge(.0, nullptr)}
    })));
}

QMDDGate gate::fSim(double theta, double phi) {
    shared_ptr<QMDDNode> fSimNode1, fSimNode2, fSimNode3, fSimNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            fSimNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(cos(theta), nullptr)}
            });
        }

        #pragma omp section
        {
            fSimNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            fSimNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            fSimNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * phi) * mathUtils::sec(theta), nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fSimNode1), QMDDEdge(-i * sin(theta), fSimNode2)},
        {QMDDEdge(-i * sin(theta), fSimNode3), QMDDEdge(cos(theta), fSimNode4)}
    })));
}

QMDDGate gate::G(double theta) {
    shared_ptr<QMDDNode> gNode1, gNode2, gNode3, gNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            gNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(cos(theta), nullptr)}
            });
        }

        #pragma omp section
        {
            gNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            gNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            gNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0 * mathUtils::sec(theta), nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, gNode1), QMDDEdge(-sin(theta), gNode2)},
        {QMDDEdge(sin(theta), gNode3), QMDDEdge(cos(theta), gNode4)}
    })));
}

QMDDGate gate::M() {
    shared_ptr<QMDDNode> mNode1, mNode2, mNode3, mNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            mNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            mNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)}
            });
        }

        #pragma omp section
        {
            mNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)}
            });
        }

        #pragma omp section
        {
            mNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, mNode1), QMDDEdge(i, mNode2)},
        {QMDDEdge(1.0, mNode3), QMDDEdge(i, mNode4)}
    })));
}

QMDDGate gate::syc() {
    shared_ptr<QMDDNode> sycNode1, sycNode2, sycNode3, sycNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            sycNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            sycNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            sycNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            sycNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, sycNode1), QMDDEdge(-i, sycNode2)},
        {QMDDEdge(-i, sycNode3), QMDDEdge(exp(-i * M_PI / 6.0), sycNode4)}
    })));
}

QMDDGate gate::CZS(double theta, double phi, double gamma) {
    shared_ptr<QMDDNode> czsNode1, czsNode2, czsNode3, czsNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            czsNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(-exp(i * gamma) * std::pow(sin(theta / 2.0), 2) + std::pow(cos(theta / 2.0), 2), nullptr)}
            });
        }

        #pragma omp section
        {
            czsNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            czsNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            czsNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(-exp(i * gamma) / (-exp(i * gamma) * std::pow(cos(theta / 2.0), 2) + std::pow(sin(theta / 2.0), 2)), nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, czsNode1), QMDDEdge((1.0 + exp(i * gamma)) / 2.0 * exp(-i * phi) * sin(theta), czsNode2)},
        {QMDDEdge((1.0 + exp(i * gamma)) / 2.0 * exp(i * phi) * sin(theta), czsNode3), QMDDEdge(-exp(i * gamma) * std::pow(cos(theta / 2.0), 2) + std::pow(sin(theta / 2.0), 2), czsNode4)}
    })));
}

QMDDGate gate::D(double theta) {
    shared_ptr<QMDDNode> dNode1, dNode2, dNode3, dNode4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            dNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
            });
        }

        #pragma omp section
        {
            dNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(-i * tan(theta), nullptr)},
                {QMDDEdge(-i * tan(theta), nullptr), QMDDEdge(1.0, nullptr)}
            });

            dNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(i * cos(theta), dNode2)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, dNode3)}
    })));
}

QMDDGate gate::RCCX() {
    shared_ptr<QMDDNode> rccxNode1, rccxNode2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            rccxNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
            });
        }

        #pragma omp section
        {
            rccxNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::Z().getStartNode())), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, rccxNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, rccxNode2)}
    })));
}

QMDDGate gate::PG() {
    shared_ptr<QMDDNode> pgNode1, pgNode2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            pgNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
            });
        }

        #pragma omp section
        {
            pgNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))},
                {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)}
            });
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, pgNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, pgNode2)}
    })));
}

QMDDGate gate::Toff() {
    shared_ptr<QMDDNode> toffNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, toffNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::CX1().getStartNode()))}
    })));
}

QMDDGate gate::fFredkin() {
    shared_ptr<QMDDNode> fFredkinNode1, fFredkinNode2, fFredkinNode3, fFredkinNode4, fFredkinNode5, fFredkinNode6;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            fFredkinNode1 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
            });
        }

        #pragma omp section
        {
            fFredkinNode2 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            fFredkinNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            fFredkinNode4 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            });
        }

        #pragma omp section
        {
            fFredkinNode5 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
            });
        }
    }

    fFredkinNode6 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fFredkinNode2), QMDDEdge(1.0, fFredkinNode3)},
        {QMDDEdge(1.0, fFredkinNode4), QMDDEdge(-1.0, fFredkinNode5)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fFredkinNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, fFredkinNode6)}
    })));
}


// matrix Rotate(const ex &k){
//     return matrix{
//         {1, 0},
//         {0, exp((2 * Pi * I) / pow(2, k))}
//     };
// }

// matrix U1(const ex &lambda){
//     return matrix{
//         {1, 0},
//         {0, exp(I * lambda)}
//     };
// }

// matrix U2(const ex &phi, const ex &lambda){
//     return matrix{
//         {1, -exp(I * lambda)},
//         {exp(I * phi), exp(I * (lambda + phi))}
//     };
// }