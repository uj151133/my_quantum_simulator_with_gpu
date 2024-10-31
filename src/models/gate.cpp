#include "gate.hpp"

QMDDGate gate::I() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeOne}
    })));
}

QMDDGate gate::Ph(double delta) {
    return QMDDGate(QMDDEdge(exp(i * delta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeOne}
    })));
}

QMDDGate gate::X() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {edgeOne, edgeZero}
    })));
}

// const QMDDGate gate::PLUS_X_GATE = [] {
//     complex<double> plusXWeight = 1 / M_SQRT2;
//     QMDDEdge plusXNode = make_shared<QMDDNode>(4);

//     plusXNode->children[0] = QMDDEdge(1, nullptr);
//     plusXNode->children[1] = QMDDEdge(i, nullptr);
//     plusXNode->children[2] = QMDDEdge(i, nullptr);
//     plusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusXEdge(plusXWeight, plusXNode);
//     return QMDDGate(plusXEdge);
// }();

// const QMDDGate gate::MINUS_X_GATE = [] {
//     complex<double> minusXWeight = 1 / M_SQRT2;
//     QMDDEdge minusXNode = make_shared<QMDDNode>(4);

//     minusXNode->children[0] = QMDDEdge(1, nullptr);
//     minusXNode->children[1] = QMDDEdge(-i, nullptr);
//     minusXNode->children[2] = QMDDEdge(-i, nullptr);
//     minusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge minusXEdge(minusXWeight, minusXNode);
//     return QMDDGate(minusXEdge);
// }();

QMDDGate gate::Y() {
    return QMDDGate(QMDDEdge(-i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, edgeOne},
        {QMDDEdge(-1.0, nullptr), edgeZero}
    })));
}

// QMDDGate createPlusYGate() {
//     complex<double> plusYWeight = 1 / M_SQRT2;
//     QMDDNode* plusYNode = new QMDDNode(4);

//     plusYNode->children[0] = QMDDEdge(1, nullptr);
//     plusYNode->children[1] = QMDDEdge(1, nullptr);
//     plusYNode->children[2] = QMDDEdge(-1, nullptr);
//     plusYNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusYEdge(plusYWeight, plusYNode);
//     return QMDDGate(plusYEdge);
// }

// QMDDGate createMinusYGate() {
//     complex<double> minusYWeight = 1 / M_SQRT2;
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
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(-1.0, nullptr)}
    })));
}

QMDDGate gate::S() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(i, nullptr)}
    })));
}

QMDDGate gate::Sdagger() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(-i, nullptr)}
    })));
}

QMDDGate gate::V() {
    QMDDEdge vEdge = QMDDEdge(i, nullptr);

    return QMDDGate(QMDDEdge(1.0 / 2.0 + i / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, vEdge},
        {vEdge, edgeOne}
    })));
}

QMDDGate gate::H() {
    return QMDDGate(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeOne},
        {edgeOne, QMDDEdge(-1.0, nullptr)}
    })));
}

QMDDGate gate::CX1() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge() , edgeZero},
        {edgeZero, gate::X().getInitialEdge()}
    })));
}

QMDDGate gate::CX2() {
    QMDDEdge cx2Edge1, cx2Edge2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            cx2Edge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            cx2Edge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeZero, edgeOne}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {cx2Edge1, cx2Edge2},
        {cx2Edge2, cx2Edge1}
    })));
}

QMDDGate gate::varCX() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::X().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::CZ() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::Z().getInitialEdge()}
    })));
}

QMDDGate gate::DCNOT() {
    QMDDEdge dcnotEdge1, dcnotEdge2, dcnotEdge3, dcnotEdge4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            dcnotEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            dcnotEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            dcnotEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeZero, edgeOne}
            }));
        }

        #pragma omp section
        {
            dcnotEdge4 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dcnotEdge1, dcnotEdge2},
        {dcnotEdge3, dcnotEdge4}
    })));
}

QMDDGate gate::SWAP(bool primitive) {
    if (primitive) {
        return QMDDGate(mathUtils::mul(mathUtils::mul(gate::CX1().getInitialEdge(), gate::CX2().getInitialEdge()), gate::CX1().getInitialEdge()));
    } else {
        QMDDEdge swapEdge1, swapEdge2, swapEdge3, swapEdge4;
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                swapEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                    {edgeOne, edgeZero},
                    {edgeZero, edgeZero}
                }));
            }

            #pragma omp section
            {
                swapEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                    {edgeZero, edgeZero},
                    {edgeOne, edgeZero}
                }));
            }

            #pragma omp section
            {
                swapEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                    {edgeZero, edgeOne},
                    {edgeZero, edgeZero}
                }));
            }

            #pragma omp section
            {
                swapEdge4 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                    {edgeZero, edgeZero},
                    {edgeZero, edgeOne}
                }));
            }
        }

        #pragma omp barrier

        return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {swapEdge1, swapEdge2},
            {swapEdge3, swapEdge4}
        })));
    }
}

QMDDGate gate::iSWAP() {
    QMDDEdge iswapEdge1, iswapEdge2, iswapEdge3, iswapEdge4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            iswapEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            iswapEdge2 = QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            iswapEdge3 = QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            iswapEdge4 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeZero, edgeOne}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {iswapEdge1, iswapEdge2},
        {iswapEdge3, iswapEdge4}
    })));
}

QMDDGate gate::P(double phi) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * phi), nullptr)}
    })));
}

QMDDGate gate::T() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * M_PI_4), nullptr)}
    })));
}

QMDDGate gate::Tdagger() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(-i * M_PI_4), nullptr)}
    })));
}

QMDDGate gate::CP(double phi) {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::P(phi).getInitialEdge()}
    })));
}

QMDDGate gate::CS() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::S().getInitialEdge()}
    })));
}

QMDDGate gate::Rx(double theta) {
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i * tanThetaHalf, nullptr)},
        {QMDDEdge(-i * tanThetaHalf, nullptr), edgeOne}
    })));
}

QMDDGate gate::Ry(double theta) {
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-tanThetaHalf, nullptr)},
        {QMDDEdge(tanThetaHalf, nullptr), edgeOne}
    })));
}

QMDDGate gate::Rz(double theta) {
    return QMDDGate(QMDDEdge(exp(-i * theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * theta), nullptr)}
    })));
}

QMDDGate gate::Rxx(double phi) {
    double phiHalf = phi / 2.0;
    double tanPhiHalf = tan(phiHalf);

    return QMDDGate(QMDDEdge(cos(phiHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), QMDDEdge(-i * tanPhiHalf, make_shared<QMDDNode>(*gate::X().getStartNode()))},
        {QMDDEdge(-i * tanPhiHalf, make_shared<QMDDNode>(*gate::X().getStartNode())), gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::Ryy(double phi) {
    double phiHalf = phi / 2.0;
    double tanPhiHalf = tan(phiHalf);

    return QMDDGate(QMDDEdge(cos(phiHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), QMDDEdge(i * tanPhiHalf, make_shared<QMDDNode>(*gate::Y().getStartNode()))},
        {QMDDEdge(-i * tanPhiHalf, make_shared<QMDDNode>(*gate::Y().getStartNode())), gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::Rzz(double phi) {
    return QMDDGate(QMDDEdge(exp(-i * phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::P(phi).getInitialEdge(), edgeZero},
        {edgeZero, QMDDEdge(exp(i * phi), make_shared<QMDDNode>(*gate::P(-phi).getStartNode()))}
    })));
}

QMDDGate gate::Rxy(double phi) {
    QMDDEdge rxyEdge1, rxyEdge2, rxyEdge3, rxyEdge4;
    double phiHalf = phi / 2.0;
    double sinPhiHalf = sin(phiHalf);
    double cosPhiHalf = cos(phiHalf);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            rxyEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(cosPhiHalf, nullptr)}
            }));
        }

        #pragma omp section
        {
            rxyEdge2 = QMDDEdge(-i * sinPhiHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            rxyEdge3 = QMDDEdge(-i * sinPhiHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            rxyEdge4 = QMDDEdge(cosPhiHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(1.0 * mathUtils::sec(phiHalf), nullptr)}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {rxyEdge1, rxyEdge2},
        {rxyEdge3, rxyEdge4}
    })));
}

QMDDGate gate::SquareSWAP() {
    QMDDEdge squareSWAPEdge1, squareSWAPEdge2, squareSWAPEdge3, squareSWAPEdge4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            squareSWAPEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge((1.0 + i) / 2.0, nullptr)}
            }));
        }

        #pragma omp section
        {
            squareSWAPEdge2 = QMDDEdge((1.0 - i) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            squareSWAPEdge3 = QMDDEdge((1.0 - i) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            squareSWAPEdge4 = QMDDEdge((1.0 + i) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(1.0 - i, nullptr)}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {squareSWAPEdge1, squareSWAPEdge2},
        {squareSWAPEdge3, squareSWAPEdge4}
    })));
}

QMDDGate gate::SquareiSWAP() {
    QMDDEdge squareiSWAPEdge1, squareiSWAPEdge2, squareiSWAPEdge3, squareiSWAPEdge4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            squareiSWAPEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(1.0 / M_SQRT2, nullptr)}
            }));
        }

        #pragma omp section
        {
            squareiSWAPEdge2 = QMDDEdge(i / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            squareiSWAPEdge3 = QMDDEdge(i / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            squareiSWAPEdge4 = QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(M_SQRT2, nullptr)}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {squareiSWAPEdge1, squareiSWAPEdge2},
        {squareiSWAPEdge3, squareiSWAPEdge4}
    })));
}

QMDDGate gate::SWAPalpha(double alpha) {
    QMDDEdge SWAPalphaEdge1, SWAPalphaEdge2, SWAPalphaEdge3, SWAPalphaEdge4;
    complex<double> expIPiAlpha = exp(i * M_PI * alpha);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            SWAPalphaEdge1 =  QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge((1.0 + expIPiAlpha) / 2.0, nullptr)}
            }));
        }

        #pragma omp section
        {
            SWAPalphaEdge2 = QMDDEdge((1.0 - expIPiAlpha) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            SWAPalphaEdge3 = QMDDEdge((1.0 - expIPiAlpha) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            SWAPalphaEdge4 = QMDDEdge((1.0 + expIPiAlpha) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(2.0 / (1.0 + expIPiAlpha), nullptr)}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {SWAPalphaEdge1, SWAPalphaEdge2},
        {SWAPalphaEdge3, SWAPalphaEdge4}
    })));
}

QMDDGate gate::FREDKIN() {
    QMDDEdge fredkinEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fredkinEdge1, edgeZero},
        {edgeZero, gate::SWAP().getInitialEdge()}
    })));
}

QMDDGate gate::U(double theta, double phi, double lambda) {
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-exp(i * lambda) * tanThetaHalf, nullptr)},
        {QMDDEdge(exp(i * phi) * tanThetaHalf, nullptr), QMDDEdge(exp(i * (lambda + phi)), nullptr)}
    })));
}

QMDDGate gate::BARENCO(double alpha, double phi, double theta) {
    double tanTheta = tan(theta);

    QMDDEdge barencoEdge1 = QMDDEdge(exp(i * alpha) * cos(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i * exp(-i * phi) * tanTheta, nullptr)},
        {QMDDEdge(-i * exp(i * phi) * tanTheta, nullptr), edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, barencoEdge1}
    })));
}

QMDDGate gate::B() {
    QMDDEdge bEdge1, bEdge2, bEdge3, bEdge4;
    double oneEighthPi = M_PI / 8.0;
    double threeEighthsPi = 3.0 * oneEighthPi;
    double sinThreeEighthsPi = sin(threeEighthsPi);
    double cosThreeEighthsPi = cos(threeEighthsPi);
    double cosOneEighthPi = cos(oneEighthPi);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            bEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(cosThreeEighthsPi * mathUtils::sec(oneEighthPi), nullptr)}
            }));
        }

        #pragma omp section
        {
            bEdge2 = QMDDEdge(i * tan(oneEighthPi), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {QMDDEdge(sinThreeEighthsPi * mathUtils::csc(oneEighthPi), nullptr), edgeZero}
            }));
        }

        #pragma omp section
        {
            bEdge3 = QMDDEdge(i * sinThreeEighthsPi / cosOneEighthPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {QMDDEdge(sin(oneEighthPi) * mathUtils::csc(threeEighthsPi), nullptr), edgeZero}
            }));
        }

        #pragma omp section
        {
            bEdge4 = QMDDEdge(cosThreeEighthsPi / cosOneEighthPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(cosOneEighthPi * mathUtils::sec(threeEighthsPi), nullptr)}
            }));
        }
    }

    return QMDDGate(QMDDEdge(cosOneEighthPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        { bEdge1, bEdge2},
        { bEdge3, bEdge4}
    })));
}

QMDDGate gate::CSX() {
    complex<double> expMinusIPiHalf = exp(i * M_PI_4);

    QMDDEdge csxEdge1 = QMDDEdge(exp(i * M_PI_4), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(expMinusIPiHalf, nullptr)},
        {QMDDEdge(expMinusIPiHalf, nullptr), edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, csxEdge1}
    })));
}

QMDDGate gate::N(double a, double b, double c) {
    QMDDEdge nEdge1, nEdge2, nEdge3, nEdge4;
    double cosAPlusB = cos(a + b);
    double cosAMinusB = cos(a - b);
    double secAMinusB = mathUtils::sec(a - b);
    complex<double> exp2IC = exp(2.0 * i * c);
    complex<double> expMinus2IC = exp(-2.0 * i * c);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            nEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(expMinus2IC * cosAPlusB * secAMinusB, nullptr)}
            }));
        }

        #pragma omp section
        {
            nEdge2 = QMDDEdge(i * tan(a - b), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {QMDDEdge(expMinus2IC * sin(a + b) * mathUtils::csc(a - b), nullptr), edgeZero}
            }));
        }

        #pragma omp section
        {
            nEdge3 = QMDDEdge(i * expMinus2IC * sin(a + b) * secAMinusB, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {QMDDEdge(exp2IC * sin(a - b) * mathUtils::csc(a + b), nullptr), edgeZero}
            }));
        }

        #pragma omp section
        {
            nEdge4 = QMDDEdge(expMinus2IC * cosAPlusB * secAMinusB, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(exp2IC * cosAMinusB * mathUtils::sec(a + b), nullptr)}
            }));
        }
    }

    return QMDDGate(QMDDEdge(exp(i * c) * cosAMinusB, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {nEdge1, nEdge2},
        {nEdge3, nEdge4}
    })));
}

QMDDGate gate::DB() {
    QMDDEdge dbEdge1, dbEdge2, dbEdge3, dbEdge4;
    double threeEighthsPi = 3.0 * M_PI / 8.0;
    double sinThreeEighthsPi = sin(threeEighthsPi);
    double cosThreeEighthsPi = cos(threeEighthsPi);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            dbEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(cosThreeEighthsPi, nullptr)}
            }));
        }

        #pragma omp section
        {
            dbEdge2 = QMDDEdge(-i * sinThreeEighthsPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            dbEdge3 = QMDDEdge(-i * sinThreeEighthsPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            dbEdge4 = QMDDEdge(cosThreeEighthsPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(1.0 * mathUtils::sec(threeEighthsPi), nullptr)}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dbEdge1, dbEdge2},
        {dbEdge3, dbEdge4}
    })));
}

QMDDGate gate::ECR() {
    QMDDEdge ecrEdge1, ecrEdge2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            ecrEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, QMDDEdge(i, nullptr)},
                {QMDDEdge(i, nullptr), edgeOne}
            }));
        }

        #pragma omp section
        {
            ecrEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, QMDDEdge(-i, nullptr)},
                {QMDDEdge(-i, nullptr), edgeOne}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, ecrEdge1},
        {ecrEdge2, edgeZero}
    })));
}

QMDDGate gate::fSim(double theta, double phi) {
    QMDDEdge fSimEdge1, fSimEdge2, fSimEdge3, fSimEdge4;
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            fSimEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(cosTheta, nullptr)}
            }));
        }

        #pragma omp section
        {
            fSimEdge2 = QMDDEdge(-i * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            fSimEdge3 = QMDDEdge(-i * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            fSimEdge4 =QMDDEdge(cosTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(exp(i * phi) * mathUtils::sec(theta), nullptr)}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fSimEdge1, fSimEdge2},
        {fSimEdge3, fSimEdge4}
    })));
}

QMDDGate gate::G(double theta) {
    QMDDEdge gEdge1, gEdge2, gEdge3, gEdge4;
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            gEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(cosTheta, nullptr)}
            }));
        }

        #pragma omp section
        {
            gEdge2 = QMDDEdge(-sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            gEdge3 = QMDDEdge(sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            gEdge4 = QMDDEdge(cosTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(1.0 * mathUtils::sec(theta), nullptr)}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gEdge1, gEdge2},
        { gEdge3, gEdge4}
    })));
}

QMDDGate gate::M() {
    QMDDEdge mEdge1, mEdge2, mEdge3, mEdge4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            mEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, QMDDEdge(i, nullptr)},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            mEdge2 = QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, QMDDEdge(-i, nullptr)}
            }));
        }

        #pragma omp section
        {
            mEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, QMDDEdge(-i, nullptr)}
            }));
        }

        #pragma omp section
        {
            mEdge4 = QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, QMDDEdge(i, nullptr)},
                {edgeZero, edgeZero}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {mEdge1, mEdge2},
        {mEdge3, mEdge4}
    })));
}

QMDDGate gate::syc() {
    QMDDEdge sycEdge1, sycEdge2, sycEdge3, sycEdge4;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            sycEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            sycEdge2 = QMDDEdge(-i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            sycEdge3 = QMDDEdge(-i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            sycEdge4 = QMDDEdge(exp(-i * M_PI / 6.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeZero, edgeOne}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {sycEdge1, sycEdge2},
        {sycEdge3, sycEdge4}
    })));
}

QMDDGate gate::CZS(double theta, double phi, double gamma) {
    QMDDEdge czsEdge1, czsEdge2, czsEdge3, czsEdge4;
    double sinTheta = sin(theta);
    double sinThetaHalf = sin(theta / 2.0);
    double cosThetaHalf = cos(theta / 2.0);
    double powSinThetaHalf = std::pow(sinThetaHalf, 2);
    double powCosThetaHalf = std::pow(cosThetaHalf, 2);
    complex<double> expIGamma = exp(i * gamma);
    complex<double> expIPhi = exp(i * phi);
    complex<double> expMinusIPhi = exp(-i * phi);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            czsEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(-expIGamma * powSinThetaHalf + powCosThetaHalf, nullptr)}
            }));
        }

        #pragma omp section
        {
            czsEdge2 = QMDDEdge((1.0 + expIGamma) / 2.0 * expMinusIPhi * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            czsEdge3 = QMDDEdge((1.0 + expIGamma) / 2.0 * expIPhi * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            czsEdge4 = QMDDEdge(-expIGamma * powCosThetaHalf + powSinThetaHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, QMDDEdge(-expIGamma / (-expIGamma * powCosThetaHalf + powSinThetaHalf), nullptr)}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {czsEdge1, czsEdge2},
        {czsEdge3, czsEdge4}
    })));
}

QMDDGate gate::D(double theta) {
    QMDDEdge dEdge1, dEdge2, dEdge3, dEdge4;
    double tanTheta = tan(theta);
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            dEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {gate::I().getInitialEdge(), edgeZero},
                {edgeZero, gate::I().getInitialEdge()}
            }));
        }

        #pragma omp section
        {
            dEdge2 = QMDDEdge(i * cos(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, QMDDEdge(-i * tanTheta, nullptr)},
                {QMDDEdge(-i * tanTheta, nullptr), edgeOne}
            }));

            dEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {gate::I().getInitialEdge(), edgeZero},
                {edgeZero, dEdge2}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dEdge1, edgeZero},
        {edgeZero, dEdge3}
    })));
}

QMDDGate gate::RCCX() {
    QMDDEdge rccxEdge1, rccxEdge2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            rccxEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {gate::I().getInitialEdge(), edgeZero},
                {edgeZero, gate::I().getInitialEdge()}
            }));
        }

        #pragma omp section
        {
            rccxEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {gate::Z().getInitialEdge(), edgeZero},
                {edgeZero, gate::X().getInitialEdge()}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {rccxEdge1, edgeZero},
        {edgeZero, rccxEdge2}
    })));
}

QMDDGate gate::PG() {
    QMDDEdge pgEdge1, pgEdge2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            pgEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {gate::I().getInitialEdge(), edgeZero},
                {edgeZero, gate::I().getInitialEdge()}
            }));
        }

        #pragma omp section
        {
            pgEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, gate::X().getInitialEdge()},
                {gate::I().getInitialEdge(), edgeZero}
            }));
        }
    }

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {pgEdge1, edgeZero},
        {edgeZero,  pgEdge2}
    })));
}

QMDDGate gate::Toff() {
    QMDDEdge toffEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {toffEdge1, edgeZero},
        {edgeZero, gate::CX1().getInitialEdge()}
    })));
}

QMDDGate gate::fFredkin() {
    QMDDEdge fFredkinEdge1, fFredkinEdge2, fFredkinEdge3, fFredkinEdge4, fFredkinEdge5, fFredkinEdge6;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            fFredkinEdge1 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {gate::I().getInitialEdge(), edgeZero},
                {edgeZero, gate::I().getInitialEdge()}
            }));
        }

        #pragma omp section
        {
            fFredkinEdge2 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            fFredkinEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            }));
        }

        #pragma omp section
        {
            fFredkinEdge4 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            }));
        }

        #pragma omp section
        {
            fFredkinEdge5 = QMDDEdge(-1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeZero, edgeOne}
            }));
        }
    }

    fFredkinEdge6 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fFredkinEdge2, fFredkinEdge3},
        {fFredkinEdge4, fFredkinEdge5}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fFredkinEdge1, edgeZero},
        {edgeZero, fFredkinEdge6}
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