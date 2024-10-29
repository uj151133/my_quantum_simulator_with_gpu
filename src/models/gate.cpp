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
//     complex<double> plusXWeight = 1 / sqrt(2.0);
//     QMDDEdge plusXNode = make_shared<QMDDNode>(4);

//     plusXNode->children[0] = QMDDEdge(1, nullptr);
//     plusXNode->children[1] = QMDDEdge(i, nullptr);
//     plusXNode->children[2] = QMDDEdge(i, nullptr);
//     plusXNode->children[3] = QMDDEdge(1, nullptr);

//     QMDDEdge plusXEdge(plusXWeight, plusXNode);
//     return QMDDGate(plusXEdge);
// }();

// const QMDDGate gate::MINUS_X_GATE = [] {
//     complex<double> minusXWeight = 1 / sqrt(2.0);
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
    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
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
            })))
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
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * phi), nullptr)}
    })));
}

QMDDGate gate::T() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * M_PI / 4.0), nullptr)}
    })));
}

QMDDGate gate::Tdagger() {
    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(-i * M_PI / 4.0), nullptr)}
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
    return QMDDGate(QMDDEdge(cos(theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i * tan(theta / 2.0), nullptr)},
        {QMDDEdge(-i * tan(theta / 2.0), nullptr), edgeOne}
    })));
}

QMDDGate gate::Ry(double theta) {
    return QMDDGate(QMDDEdge(cos(theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-tan(theta / 2.0), nullptr)},
        {QMDDEdge(tan(theta / 2.0), nullptr), edgeOne}
    })));
}

QMDDGate gate::Rz(double theta) {
    return QMDDGate(QMDDEdge(exp(-i * theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, edgeZero},
        {edgeZero, QMDDEdge(exp(i * theta), nullptr)}
    })));
}

QMDDGate gate::Rxx(double phi) {
    return QMDDGate(QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), QMDDEdge(-i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::X().getStartNode()))},
        {QMDDEdge(-i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::X().getStartNode())), gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::Ryy(double phi) {
    return QMDDGate(QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), QMDDEdge(i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::Y().getStartNode()))},
        {QMDDEdge(-i * tan(phi / 2.0), make_shared<QMDDNode>(*gate::Y().getStartNode())), gate::I().getInitialEdge()}
    })));
}

QMDDGate gate::Rzz(double phi) {
    return QMDDGate(QMDDEdge(exp(-i * phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::P(phi).getInitialEdge(), edgeZero},
        {edgeZero, QMDDEdge(exp(i * phi), make_shared<QMDDNode>(*gate::P(-phi).getStartNode()))}
    })));
}

QMDDGate gate::Rxy(double phi) {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, phi]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cos(phi / 2.0), nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, phi]() {
        promise2.set_value(QMDDEdge(-i * sin(phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, phi]() {
        promise3.set_value(QMDDEdge(-i * sin(phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, phi]() {
        promise4.set_value(QMDDEdge(cos(phi / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(1.0 * mathUtils::sec(phi / 2.0), nullptr)}
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

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(1.0 / sqrt(2.0), nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(i / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge(i / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(sqrt(2.0), nullptr)}
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

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, alpha]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge((1.0 - exp(i * M_PI * alpha)) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, alpha]() {
        promise4.set_value(QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(2.0 / (1.0 + exp(i * M_PI * alpha)), nullptr)}
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
    return QMDDGate(QMDDEdge(cos(theta / 2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-exp(i * lambda) * tan(theta / 2.0), nullptr)},
        {QMDDEdge(exp(i * phi) * tan(theta / 2.0), nullptr), QMDDEdge(exp(i * (lambda + phi)), nullptr)}
    })));
}

QMDDGate gate::BARENCO(double alpha, double phi, double theta) {
    QMDDEdge barencoEdge1 = QMDDEdge(exp(i * alpha) * cos(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(-i * exp(-i * phi) * tan(theta), nullptr)},
        {QMDDEdge(-i * exp(i * phi) * tan(theta), nullptr), edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, barencoEdge1}
    })));
}

QMDDGate gate::B() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cos(3.0 * M_PI / 8.0) * mathUtils::sec(M_PI / 8.0), nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(i * tan(M_PI / 8.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {QMDDEdge(sin(3.0 * M_PI / 8.0) * mathUtils::csc(M_PI / 8.0), nullptr), edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge(i * sin(3.0 * M_PI / 8.0) / cos(M_PI / 8.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {QMDDEdge(sin(M_PI / 8.0) * mathUtils::csc(3.0 * M_PI / 8.0), nullptr), edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(QMDDEdge(cos(3.0 * M_PI / 8.0) / cos(M_PI / 8.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cos(M_PI / 8.0) * mathUtils::sec(3.0 * M_PI / 8.0), nullptr)}
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

    return QMDDGate(QMDDEdge(cos(M_PI / 8.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        { bEdge1, bEdge2},
        { bEdge3, bEdge4}
    })));
}

QMDDGate gate::CSX() {
    QMDDEdge csxEdge1 = QMDDEdge(exp(i * M_PI / 4.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeOne, QMDDEdge(exp(-i * M_PI / 2.0), nullptr)},
        {QMDDEdge(exp(-i * M_PI / 2.0), nullptr), edgeOne}
    }));

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {gate::I().getInitialEdge(), edgeZero},
        {edgeZero, csxEdge1}
    })));
}

QMDDGate gate::N(double a, double b, double c) {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, a, b, c]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(exp(-2.0 * i * c) * cos(a + b) * mathUtils::sec(a - b), nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, a, b, c]() {
        promise2.set_value(QMDDEdge(i * tan(a - b), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {QMDDEdge(exp(-2.0 * i * c) * sin(a + b) * mathUtils::csc(a - b), nullptr), edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, a, b, c]() {
        promise3.set_value(QMDDEdge(i * exp(-2.0 * i * c) * sin(a + b) * mathUtils::sec(a - b), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {QMDDEdge(exp(2.0 * i * c) * sin(a - b) * mathUtils::csc(a + b), nullptr), edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, a, b, c]() {
        promise4.set_value(QMDDEdge(exp(-2.0 * i * c) * cos(a + b) * mathUtils::sec(a - b), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(exp(2.0 * i * c) * cos(a - b) * mathUtils::sec(a + b), nullptr)}
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

    return QMDDGate(QMDDEdge(exp(i * c) * cos(a - b), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {nEdge1, nEdge2},
        {nEdge3, nEdge4}
    })));
}

QMDDGate gate::DB() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cos(3.0 * M_PI / 8.0), nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(QMDDEdge(-i * sin(3.0 * M_PI / 8.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(QMDDEdge(-i * sin(3.0 * M_PI / 8.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(QMDDEdge(cos(3.0 * M_PI / 8.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(1.0 * mathUtils::sec(3.0 * M_PI / 8.0), nullptr)}
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

    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {edgeZero, ecrEdge1},
        {ecrEdge2, edgeZero}
    })));
}

QMDDGate gate::fSim(double theta, double phi) {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, theta]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cos(theta), nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, theta]() {
        promise2.set_value(QMDDEdge(-i * sin(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, theta]() {
        promise3.set_value(QMDDEdge(-i * sin(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, phi, theta]() {
        promise4.set_value(QMDDEdge(cos(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
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

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, theta]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(cos(theta), nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, theta]() {
        promise2.set_value(QMDDEdge(-sin(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, theta]() {
        promise3.set_value(QMDDEdge(sin(theta),make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, theta]() {
        promise4.set_value(QMDDEdge(cos(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
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

    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {mEdge1, mEdge2},
        {mEdge3, mEdge4}
    })));
}

QMDDGate gate::syc() {

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

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<QMDDEdge> promise1, promise2, promise3, promise4;
    boost::fibers::future<QMDDEdge> future1 = promise1.get_future();
    boost::fibers::future<QMDDEdge> future2 = promise2.get_future();
    boost::fibers::future<QMDDEdge> future3 = promise3.get_future();
    boost::fibers::future<QMDDEdge> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, theta, gamma]() {
        promise1.set_value(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(-exp(i * gamma) * std::pow(sin(theta / 2.0), 2) + std::pow(cos(theta / 2.0), 2), nullptr)}
        })));
    }).detach();

    boost::fibers::fiber([&promise2, theta, phi, gamma]() {
        promise2.set_value(QMDDEdge((1.0 + exp(i * gamma)) / 2.0 * exp(-i * phi) * sin(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeZero},
            {edgeOne, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise3, theta, phi, gamma]() {
        promise3.set_value(QMDDEdge((1.0 + exp(i * gamma)) / 2.0 * exp(i * phi) * sin(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeZero, edgeOne},
            {edgeZero, edgeZero}
        })));
    }).detach();

    boost::fibers::fiber([&promise4, theta, gamma]() {
        promise4.set_value(QMDDEdge(-exp(i * gamma) * std::pow(cos(theta / 2.0), 2) + std::pow(sin(theta / 2.0), 2), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, edgeZero},
            {edgeZero, QMDDEdge(-exp(i * gamma) / (-exp(i * gamma) * std::pow(cos(theta / 2.0), 2) + std::pow(sin(theta / 2.0), 2)), nullptr)}
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

    boost::fibers::fiber([&promise2, theta]() {
        promise2.set_value(QMDDEdge(i * cos(theta), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {edgeOne, QMDDEdge(-i * tan(theta), nullptr)},
            {QMDDEdge(-i * tan(theta), nullptr), edgeOne}
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