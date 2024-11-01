#include "gate.hpp"


QMDDGate gate::I() {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeOne}
    })));
}

QMDDGate gate::Ph(double delta) {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(exp(i * delta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, edgeOne}
    })));
}

QMDDGate gate::X() {
    call_once(initFlag, init);
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
    call_once(initFlag, init);
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
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(-1.0, nullptr)}
    })));
}

QMDDGate gate::S() {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(i, nullptr)}
    })));
}

QMDDGate gate::Sdagger() {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(-i, nullptr)}
    })));
}

QMDDGate gate::V() {
    call_once(initFlag, init);
    QMDDEdge vEdge = QMDDEdge(i, nullptr);

    return QMDDGate(QMDDEdge(1.0 / 2.0 + i / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, vEdge},
        {vEdge, edgeOne}
    })));
}

QMDDGate gate::H() {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeOne},
        {edgeOne, QMDDEdge(-1.0, nullptr)}
    })));
}

QMDDGate gate::CX1() {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge() , edgeZero},
        {edgeZero, gate::X().getInitialEdge()}
    })));
}

QMDDGate gate::CX2() {
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeZero, edgeOne}
        })));
    }).detach();

    future1.wait();
    future2.wait();

    QMDDEdge cx2Edge1 = future1.get();
    QMDDEdge cx2Edge2 = future2.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {cx2Edge1, cx2Edge2},
        {cx2Edge2, cx2Edge1}
    })));
}

QMDDGate gate::varCX() {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::X().getInitialEdge(), edgeZero},
        {edgeZero, gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::CZ() {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::Z().getInitialEdge()}
    })));
}

QMDDGate gate::DCNOT() {
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeZero, edgeOne}
        })));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge dcnotEdge1 = future1.get();
    QMDDEdge dcnotEdge2 = future2.get();
    QMDDEdge dcnotEdge3 = future3.get();
    QMDDEdge dcnotEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dcnotEdge1, dcnotEdge2},
        {dcnotEdge3, dcnotEdge4}
    })));
}

QMDDGate gate::SWAP(bool primitive) {
    call_once(initFlag, init);
    if (primitive) {
        return QMDDGate(mathUtils::mul(mathUtils::mul(gate::CX1().getInitialEdge(), gate::CX2().getInitialEdge()), gate::CX1().getInitialEdge()));
    } else {

        boost::fibers::use_scheduling_algorithm<CustomScheduler>();

        boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
        boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
        boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
        boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
        boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

        boost::fibers::fiber([&promise1]() {
            promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeOne, edgeZero},
                {edgeZero, edgeZero}
            })));
        }).detach();

        boost::fibers::fiber([&promise2]() {
            promise2.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeOne, edgeZero}
            })));
        }).detach();

        boost::fibers::fiber([&promise3]() {
            promise3.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeOne},
                {edgeZero, edgeZero}
            })));
        }).detach();

        boost::fibers::fiber([&promise4]() {
            promise4.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {edgeZero, edgeZero},
                {edgeZero, edgeOne}
            })));
        }).detach();

        future1.wait();
        future2.wait();
        future3.wait();
        future4.wait();

        QMDDEdge swapEdge1 = future1.get();
        QMDDEdge swapEdge2 = future2.get();
        QMDDEdge swapEdge3 = future3.get();
        QMDDEdge swapEdge4 = future4.get();

        return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {swapEdge1, swapEdge2},
            {swapEdge3, swapEdge4}
        })));
    }
}

QMDDGate gate::iSWAP() {
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeZero, edgeOne}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge iswapEdge1 = future1.get();
    QMDDEdge iswapEdge2 = future2.get();
    QMDDEdge iswapEdge3 = future3.get();
    QMDDEdge iswapEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {iswapEdge1, iswapEdge2},
        {iswapEdge3, iswapEdge4}
    })));
}

QMDDGate gate::P(double phi) {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * phi), nullptr)}
    })));
}

QMDDGate gate::T() {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * M_PI_4), nullptr)}
    })));
}

QMDDGate gate::Tdagger() {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(-i * M_PI_4), nullptr)}
    })));
}

QMDDGate gate::CP(double phi) {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::P(phi).getInitialEdge()}
    })));
}

QMDDGate gate::CS() {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, gate::S().getInitialEdge()}
    })));
}

QMDDGate gate::Rx(double theta) {
    call_once(initFlag, init);
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i * tanThetaHalf, nullptr)},
        {QMDDEdge(-i * tanThetaHalf, nullptr), edgeOne}
    })));
}

QMDDGate gate::Ry(double theta) {
    call_once(initFlag, init);
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-tanThetaHalf, nullptr)},
        {QMDDEdge(tanThetaHalf, nullptr), edgeOne}
    })));
}

QMDDGate gate::Rz(double theta) {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(exp(-i * theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * theta), nullptr)}
    })));
}

QMDDGate gate::Rxx(double phi) {
    call_once(initFlag, init);
    double phiHalf = phi / 2.0;
    double tanPhiHalf = tan(phiHalf);

    return QMDDGate(QMDDEdge(cos(phiHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), QMDDEdge(-i * tanPhiHalf, make_shared<QMDDNode>(*gate::X().getStartNode()))},
        {QMDDEdge(-i * tanPhiHalf, make_shared<QMDDNode>(*gate::X().getStartNode())), gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::Ryy(double phi) {
    call_once(initFlag, init);
    double phiHalf = phi / 2.0;
    double tanPhiHalf = tan(phiHalf);

    return QMDDGate(QMDDEdge(cos(phiHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), QMDDEdge(i * tanPhiHalf, make_shared<QMDDNode>(*gate::Y().getStartNode()))},
        {QMDDEdge(-i * tanPhiHalf, make_shared<QMDDNode>(*gate::Y().getStartNode())), gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::Rzz(double phi) {
    call_once(initFlag, init);
    return QMDDGate(QMDDEdge(exp(-i * phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::P(phi).getInitialEdge(), edgeZero},
        {edgeZero, QMDDEdge(exp(i * phi), make_shared<QMDDNode>(*gate::P(-phi).getStartNode()))}
    })));
}

QMDDGate gate::Rxy(double phi) {
    call_once(initFlag, init);
    double phiHalf = phi / 2.0;
    double sinPhiHalf = sin(phiHalf);
    double cosPhiHalf = cos(phiHalf);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, cosPhiHalf]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cosPhiHalf, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, sinPhiHalf]() {
        promise2.set_value(QMDDEdge(-i * sinPhiHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, sinPhiHalf]() {
        promise3.set_value(QMDDEdge(-i * sinPhiHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, phiHalf, cosPhiHalf]() {
        promise4.set_value(QMDDEdge(cosPhiHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(1.0 * mathUtils::sec(phiHalf), nullptr)}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge rxyEdge1 = future1.get();
    QMDDEdge rxyEdge2 = future2.get();
    QMDDEdge rxyEdge3 = future3.get();
    QMDDEdge rxyEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {rxyEdge1, rxyEdge2},
        {rxyEdge3, rxyEdge4}
    })));
}

QMDDGate gate::SquareSWAP() {
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge((1.0 + i) / 2.0, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge((1.0 - i) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge((1.0 - i) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(QMDDEdge((1.0 + i) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(1.0 - i, nullptr)}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge squareSWAPEdge1 = future1.get();
    QMDDEdge squareSWAPEdge2 = future2.get();
    QMDDEdge squareSWAPEdge3 = future3.get();
    QMDDEdge squareSWAPEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {squareSWAPEdge1, squareSWAPEdge2},
        {squareSWAPEdge3, squareSWAPEdge4}
    })));
}

QMDDGate gate::SquareiSWAP() {
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(1.0 / M_SQRT2, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(i / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge(i / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(M_SQRT2, nullptr)}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge squareiSWAPEdge1 = future1.get();
    QMDDEdge squareiSWAPEdge2 = future2.get();
    QMDDEdge squareiSWAPEdge3 = future3.get();
    QMDDEdge squareiSWAPEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {squareiSWAPEdge1, squareiSWAPEdge2},
        {squareiSWAPEdge3, squareiSWAPEdge4}
    })));
}

QMDDGate gate::SWAPalpha(double alpha) {
    call_once(initFlag, init);
    complex<double> expIPiAlpha = exp(i * M_PI * alpha);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, expIPiAlpha]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge((1.0 + expIPiAlpha) / 2.0, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, expIPiAlpha]() {
        promise2.set_value(QMDDEdge((1.0 - expIPiAlpha) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, expIPiAlpha]() {
        promise3.set_value(QMDDEdge((1.0 - expIPiAlpha) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, expIPiAlpha]() {
        promise4.set_value(QMDDEdge((1.0 + expIPiAlpha) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(2.0 / (1.0 + expIPiAlpha), nullptr)}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge SWAPalphaEdge1 = future1.get();
    QMDDEdge SWAPalphaEdge2 = future2.get();
    QMDDEdge SWAPalphaEdge3 = future3.get();
    QMDDEdge SWAPalphaEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {SWAPalphaEdge1, SWAPalphaEdge2},
        {SWAPalphaEdge3, SWAPalphaEdge4}
    })));
}

QMDDGate gate::FREDKIN() {
    call_once(initFlag, init);
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
    call_once(initFlag, init);
    double thetaHalf = theta / 2.0;
    double tanThetaHalf = tan(thetaHalf);

    return QMDDGate(QMDDEdge(cos(thetaHalf), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-exp(i * lambda) * tanThetaHalf, nullptr)},
        {QMDDEdge(exp(i * phi) * tanThetaHalf, nullptr), QMDDEdge(exp(i * (lambda + phi)), nullptr)}
    })));
}

QMDDGate gate::BARENCO(double alpha, double phi, double theta) {
    call_once(initFlag, init);
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
    call_once(initFlag, init);
    double oneEighthPi = M_PI / 8.0;
    double threeEighthsPi = 3.0 * oneEighthPi;
    double sinThreeEighthsPi = sin(threeEighthsPi);
    double cosThreeEighthsPi = cos(threeEighthsPi);
    double cosOneEighthPi = cos(oneEighthPi);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, oneEighthPi, cosThreeEighthsPi]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cosThreeEighthsPi * mathUtils::sec(oneEighthPi), nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, oneEighthPi, sinThreeEighthsPi]() {
        promise2.set_value(QMDDEdge(i * tan(oneEighthPi), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {QMDDEdge(sinThreeEighthsPi * mathUtils::csc(oneEighthPi), nullptr), edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, oneEighthPi, threeEighthsPi, sinThreeEighthsPi, cosOneEighthPi]() {
        promise3.set_value(QMDDEdge(i * sinThreeEighthsPi / cosOneEighthPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {QMDDEdge(sin(oneEighthPi) * mathUtils::csc(threeEighthsPi), nullptr), edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, threeEighthsPi, cosOneEighthPi, cosThreeEighthsPi]() {
        promise4.set_value(QMDDEdge(cosThreeEighthsPi / cosOneEighthPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cosOneEighthPi * mathUtils::sec(threeEighthsPi), nullptr)}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge bEdge1 = future1.get();
    QMDDEdge bEdge2 = future2.get();
    QMDDEdge bEdge3 = future3.get();
    QMDDEdge bEdge4 = future4.get();

    return QMDDGate(QMDDEdge(cosOneEighthPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        { bEdge1, bEdge2},
        { bEdge3, bEdge4}
    })));
}

QMDDGate gate::CSX() {
    call_once(initFlag, init);
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
    call_once(initFlag, init);
    double cosAPlusB = cos(a + b);
    double cosAMinusB = cos(a - b);
    double secAMinusB = mathUtils::sec(a - b);
    complex<double> exp2IC = exp(2.0 * i * c);
    complex<double> expMinus2IC = exp(-2.0 * i * c);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, cosAPlusB, secAMinusB, expMinus2IC]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(expMinus2IC * cosAPlusB * secAMinusB, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, a, b, expMinus2IC]() {
        promise2.set_value(QMDDEdge(i * tan(a - b), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {QMDDEdge(expMinus2IC * sin(a + b) * mathUtils::csc(a - b), nullptr), edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, a, b, secAMinusB, exp2IC, expMinus2IC]() {
        promise3.set_value(QMDDEdge(i * expMinus2IC * sin(a + b) * secAMinusB, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {QMDDEdge(exp2IC * sin(a - b) * mathUtils::csc(a + b), nullptr), edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, a, b, cosAPlusB, cosAMinusB, secAMinusB, exp2IC, expMinus2IC]() {
        promise4.set_value(QMDDEdge(expMinus2IC * cosAPlusB * secAMinusB, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(exp2IC * cosAMinusB * mathUtils::sec(a + b), nullptr)}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge nEdge1 = future1.get();
    QMDDEdge nEdge2 = future2.get();
    QMDDEdge nEdge3 = future3.get();
    QMDDEdge nEdge4 = future4.get();

    return QMDDGate(QMDDEdge(exp(i * c) * cosAMinusB, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {nEdge1, nEdge2},
        {nEdge3, nEdge4}
    })));
}

QMDDGate gate::DB() {
    call_once(initFlag, init);
    double threeEighthsPi = 3.0 * M_PI / 8.0;
    double sinThreeEighthsPi = sin(threeEighthsPi);
    double cosThreeEighthsPi = cos(threeEighthsPi);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, cosThreeEighthsPi]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cosThreeEighthsPi, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, sinThreeEighthsPi]() {
        promise2.set_value(QMDDEdge(-i * sinThreeEighthsPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, sinThreeEighthsPi]() {
        promise3.set_value(QMDDEdge(-i * sinThreeEighthsPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, threeEighthsPi, cosThreeEighthsPi]() {
        promise4.set_value(QMDDEdge(cosThreeEighthsPi, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(1.0 * mathUtils::sec(threeEighthsPi), nullptr)}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge dbEdge1 = future1.get();
    QMDDEdge dbEdge2 = future2.get();
    QMDDEdge dbEdge3 = future3.get();
    QMDDEdge dbEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dbEdge1, dbEdge2},
        {dbEdge3, dbEdge4}
    })));
}

QMDDGate gate::ECR() {
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, QMDDEdge(i, nullptr)},
            {QMDDEdge(i, nullptr), edgeOne}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, QMDDEdge(-i, nullptr)},
            {QMDDEdge(-i, nullptr), edgeOne}
        })));
    }).detach();

    future1.wait();
    future2.wait();

    QMDDEdge ecrEdge1 = future1.get();
    QMDDEdge ecrEdge2 = future2.get();

    return QMDDGate(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, ecrEdge1},
        {ecrEdge2, edgeZero}
    })));
}

QMDDGate gate::fSim(double theta, double phi) {
    call_once(initFlag, init);
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, cosTheta]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cosTheta, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, sinTheta]() {
        promise2.set_value(QMDDEdge(-i * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, sinTheta]() {
        promise3.set_value(QMDDEdge(-i * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, phi, theta, cosTheta]() {
        promise4.set_value(QMDDEdge(cosTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(exp(i * phi) * mathUtils::sec(theta), nullptr)}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge fSimEdge1 = future1.get();
    QMDDEdge fSimEdge2 = future2.get();
    QMDDEdge fSimEdge3 = future3.get();
    QMDDEdge fSimEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {fSimEdge1, fSimEdge2},
        {fSimEdge3, fSimEdge4}
    })));
}

QMDDGate gate::G(double theta) {
    call_once(initFlag, init);
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, cosTheta]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cosTheta, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, sinTheta]() {
        promise2.set_value(QMDDEdge(-sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, sinTheta]() {
        promise3.set_value(QMDDEdge(sinTheta,make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, theta, cosTheta]() {
        promise4.set_value(QMDDEdge(cosTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(1.0 * mathUtils::sec(theta), nullptr)}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge gEdge1 = future1.get();
    QMDDEdge gEdge2 = future2.get();
    QMDDEdge gEdge3 = future3.get();
    QMDDEdge gEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gEdge1, gEdge2},
        { gEdge3, gEdge4}
    })));
}

QMDDGate gate::M() {
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, QMDDEdge(i, nullptr)},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, QMDDEdge(-i, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, QMDDEdge(-i, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(QMDDEdge(i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, QMDDEdge(i, nullptr)},
            {edgeZero, edgeZero}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge mEdge1 = future1.get();
    QMDDEdge mEdge2 = future2.get();
    QMDDEdge mEdge3 = future3.get();
    QMDDEdge mEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0 / M_SQRT2, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {mEdge1, mEdge2},
        {mEdge3, mEdge4}
    })));
}

QMDDGate gate::syc() {
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(-i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge(-i, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(QMDDEdge(exp(-i * M_PI / 6.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeZero, edgeOne}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge sycEdge1 = future1.get();
    QMDDEdge sycEdge2 = future2.get();
    QMDDEdge sycEdge3 = future3.get();
    QMDDEdge sycEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {sycEdge1, sycEdge2},
        {sycEdge3, sycEdge4}
    })));
}

QMDDGate gate::CZS(double theta, double phi, double gamma) {
    call_once(initFlag, init);
    double sinTheta = sin(theta);
    double sinThetaHalf = sin(theta / 2.0);
    double cosThetaHalf = cos(theta / 2.0);
    double powSinThetaHalf = std::pow(sinThetaHalf, 2);
    double powCosThetaHalf = std::pow(cosThetaHalf, 2);
    complex<double> expIGamma = exp(i * gamma);
    complex<double> expIPhi = exp(i * phi);
    complex<double> expMinusIPhi = exp(-i * phi);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, powSinThetaHalf, powCosThetaHalf, expIGamma]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(-expIGamma * powSinThetaHalf + powCosThetaHalf, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, sinTheta, expMinusIPhi, expIGamma]() {
        promise2.set_value(QMDDEdge((1.0 + expIGamma) / 2.0 * expMinusIPhi * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, sinTheta, expIPhi, expIGamma]() {
        promise3.set_value(QMDDEdge((1.0 + expIGamma) / 2.0 * expIPhi * sinTheta, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, powSinThetaHalf, powCosThetaHalf, expIGamma]() {
        promise4.set_value(QMDDEdge(-expIGamma * powCosThetaHalf + powSinThetaHalf, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(-expIGamma / (-expIGamma * powCosThetaHalf + powSinThetaHalf), nullptr)}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    QMDDEdge czsEdge1 = future1.get();
    QMDDEdge czsEdge2 = future2.get();
    QMDDEdge czsEdge3 = future3.get();
    QMDDEdge czsEdge4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {czsEdge1, czsEdge2},
        {czsEdge3, czsEdge4}
    })));
}

QMDDGate gate::D(double theta) {
    call_once(initFlag, init);
    double tanTheta = tan(theta);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {gate::I().getInitialEdge(), edgeZero},
            {edgeZero, gate::I().getInitialEdge()}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, theta, tanTheta]() {
        promise2.set_value(QMDDEdge(i * cos(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, QMDDEdge(-i * tanTheta, nullptr)},
            {QMDDEdge(-i * tanTheta, nullptr), edgeOne}
        })));
    }).detach();

    future1.wait();
    future2.wait();

    QMDDEdge dEdge1 = future1.get();
    QMDDEdge dEdge2 = future2.get();

    QMDDEdge dEdge3 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, dEdge2}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {dEdge1, edgeZero},
        {edgeZero, dEdge3}
    })));
}

QMDDGate gate::RCCX() {
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {gate::I().getInitialEdge(), edgeZero},
            {edgeZero, gate::I().getInitialEdge()}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {gate::Z().getInitialEdge(), edgeZero},
            {edgeZero, gate::X().getInitialEdge()}
        })));
    }).detach();

    future1.wait();
    future2.wait();

    QMDDEdge rccxEdge1 = future1.get();
    QMDDEdge rccxEdge2 = future2.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {rccxEdge1, edgeZero},
        {edgeZero, rccxEdge2}
    })));
}

QMDDGate gate::PG() {
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {gate::I().getInitialEdge(), edgeZero},
            {edgeZero, gate::I().getInitialEdge()}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, gate::X().getInitialEdge()},
            {gate::I().getInitialEdge(), edgeZero}
        })));
    }).detach();

    future1.wait();
    future2.wait();

    QMDDEdge pgEdge1 = future1.get();
    QMDDEdge pgEdge2 = future2.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {pgEdge1, edgeZero},
        {edgeZero, pgEdge2}
    })));
}

QMDDGate gate::Toff() {
    call_once(initFlag, init);
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
    call_once(initFlag, init);

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4, promise5;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();
    boost::fibers::future<QMDDEdge> future5 = promise5.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {gate::I().getInitialEdge(), edgeZero},
            {edgeZero, gate::I().getInitialEdge()}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise5]() {
        promise5.set_value(QMDDEdge(-1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeZero, edgeOne}
        })));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();
    future5.wait();

    QMDDEdge fFredkinEdge1 = future1.get();
    QMDDEdge fFredkinEdge2 = future2.get();
    QMDDEdge fFredkinEdge3 = future3.get();
    QMDDEdge fFredkinEdge4 = future4.get();
    QMDDEdge fFredkinEdge5 = future5.get();


    QMDDEdge fFredkinEdge6 = QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
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