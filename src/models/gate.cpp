#include "gate.hpp"

static complex<double> i(0.0, 1.0);

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

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();

    shared_ptr<QMDDNode> cx2Node1 = future1.get();
    shared_ptr<QMDDNode> cx2Node2 = future2.get();

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

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> dcnotNode1 = future1.get();
    shared_ptr<QMDDNode> dcnotNode2 = future2.get();
    shared_ptr<QMDDNode> dcnotNode3 = future3.get();
    shared_ptr<QMDDNode> dcnotNode4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dcnotNode1), QMDDEdge(1.0, dcnotNode2)},
        {QMDDEdge(1.0, dcnotNode3), QMDDEdge(1.0, dcnotNode4)}
    })));
}

QMDDGate gate::SWAP(bool primitive) {
    if (primitive) {
        return QMDDGate(mathUtils::mul(mathUtils::mul(gate::CX1().getInitialEdge(), gate::CX2().getInitialEdge()), gate::CX1().getInitialEdge()));
    } else {

        boost::fibers::use_scheduling_algorithm<CustomScheduler>();

        boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
        boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
        boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
        boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
        boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

        boost::fibers::fiber([&promise1]() {
            promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            }));
        }).detach();

        boost::fibers::fiber([&promise2]() {
            promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
            }));
        }).detach();

        boost::fibers::fiber([&promise3]() {
            promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
            }));
        }).detach();

        boost::fibers::fiber([&promise4]() {
            promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
                {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
                {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
            }));
        }).detach();

        future1.wait();
        future2.wait();
        future3.wait();
        future4.wait();

        shared_ptr<QMDDNode> swapNode1 = future1.get();
        shared_ptr<QMDDNode> swapNode2 = future2.get();
        shared_ptr<QMDDNode> swapNode3 = future3.get();
        shared_ptr<QMDDNode> swapNode4 = future4.get();

        return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, swapNode1), QMDDEdge(1.0, swapNode2)},
            {QMDDEdge(1.0, swapNode3), QMDDEdge(1.0, swapNode4)}
        })));
    }
}

QMDDGate gate::iSWAP() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> iswapNode1 = future1.get();
    shared_ptr<QMDDNode> iswapNode2 = future2.get();
    shared_ptr<QMDDNode> iswapNode3 = future3.get();
    shared_ptr<QMDDNode> iswapNode4 = future4.get();

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
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::P(phi).getStartNode()))}
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

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, phi]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(cos(phi / 2.0), nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4, phi]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0 * mathUtils::sec(phi / 2.0), nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> rxyNode1 = future1.get();
    shared_ptr<QMDDNode> rxyNode2 = future2.get();
    shared_ptr<QMDDNode> rxyNode3 = future3.get();
    shared_ptr<QMDDNode> rxyNode4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, rxyNode1), QMDDEdge(-i * sin(phi / 2.0), rxyNode2)},
        {QMDDEdge(-i * sin(phi / 2.0), rxyNode3), QMDDEdge(cos(phi / 2.0), rxyNode4)}
    })));
}

QMDDGate gate::SquareSWAP() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge((1.0 + i) / 2.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0 - i, nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> squareSWAPNode1 = future1.get();
    shared_ptr<QMDDNode> squareSWAPNode2 = future2.get();
    shared_ptr<QMDDNode> squareSWAPNode3 = future3.get();
    shared_ptr<QMDDNode> squareSWAPNode4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, squareSWAPNode1), QMDDEdge((1.0 - i) / 2.0, squareSWAPNode2)},
        {QMDDEdge((1.0 - i) / 2.0, squareSWAPNode3), QMDDEdge((1.0 + i) / 2.0, squareSWAPNode4)}
    })));
}

QMDDGate gate::SquareiSWAP() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0 / sqrt(2.0), nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(sqrt(2.0), nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> squareiSWAPNode1 = future1.get();
    shared_ptr<QMDDNode> squareiSWAPNode2 = future2.get();
    shared_ptr<QMDDNode> squareiSWAPNode3 = future3.get();
    shared_ptr<QMDDNode> squareiSWAPNode4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, squareiSWAPNode1), QMDDEdge(i / sqrt(2.0), squareiSWAPNode2)},
        {QMDDEdge(i / sqrt(2.0), squareiSWAPNode3), QMDDEdge(1.0 / sqrt(2.0), squareiSWAPNode4)}
    })));
}

QMDDGate gate::SWAPalpha(double alpha) {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, alpha]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge((1.0 + exp(i * M_PI * alpha)) / 2.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4, alpha]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(2.0 / (1.0 + exp(i * M_PI * alpha)), nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> SWAPalphaNode1 = future1.get();
    shared_ptr<QMDDNode> SWAPalphaNode2 = future2.get();
    shared_ptr<QMDDNode> SWAPalphaNode3 = future3.get();
    shared_ptr<QMDDNode> SWAPalphaNode4 = future4.get();

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

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(cos(3.0 * M_PI / 8.0) * mathUtils::sec(M_PI / 8.0), nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(sin(3.0 * M_PI / 8.0) * mathUtils::csc(M_PI / 8.0), nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(sin(M_PI / 8.0) * mathUtils::csc(3.0 * M_PI / 8.0), nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(cos(M_PI / 8.0) * mathUtils::sec(3.0 * M_PI / 8.0), nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> bNode1 = future1.get();
    shared_ptr<QMDDNode> bNode2 = future2.get();
    shared_ptr<QMDDNode> bNode3 = future3.get();
    shared_ptr<QMDDNode> bNode4 = future4.get();

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

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, a, b, c]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(exp(-2.0 * i * c) * cos(a + b) * mathUtils::sec(a - b), nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2, a, b, c]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(exp(-2.0 * i * c) * sin(a + b) * mathUtils::csc(a - b), nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3, a, b, c]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(exp(2.0 * i * c) * sin(a - b) * mathUtils::csc(a + b), nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4, a, b, c]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(exp(2.0 * i * c) * cos(a - b) * mathUtils::sec(a + b), nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> nNode1 = future1.get();
    shared_ptr<QMDDNode> nNode2 = future2.get();
    shared_ptr<QMDDNode> nNode3 = future3.get();
    shared_ptr<QMDDNode> nNode4 = future4.get();

    return QMDDGate(QMDDEdge(exp(i * c) * cos(a - b), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, nNode1), QMDDEdge(i * tan(a - b), nNode2)},
        {QMDDEdge(i * exp(-2.0 * i * c) * sin(a + b) * mathUtils::sec(a - b), nNode3), QMDDEdge(exp(-2.0 * i * c) * cos(a + b) * mathUtils::sec(a - b), nNode4)}
    })));
}

QMDDGate gate::DB() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(cos(3.0 * M_PI / 8.0), nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0 * mathUtils::sec(3.0 * M_PI / 8.0), nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> dbNode1 = future1.get();
    shared_ptr<QMDDNode> dbNode2 = future2.get();
    shared_ptr<QMDDNode> dbNode3 = future3.get();
    shared_ptr<QMDDNode> dbNode4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dbNode1), QMDDEdge(-i * sin(3.0 * M_PI / 8.0), dbNode2)},
        {QMDDEdge(-i * sin(3.0 * M_PI / 8.0), dbNode3), QMDDEdge(cos(3.0 * M_PI / 8.0), dbNode4)}
    })));
}

QMDDGate gate::ECR() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
            {QMDDEdge(i, nullptr), QMDDEdge(1.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)},
            {QMDDEdge(-i, nullptr), QMDDEdge(1.0, nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();

    shared_ptr<QMDDNode> ecrNode1 = future1.get();
    shared_ptr<QMDDNode> ecrNode2 = future2.get();

    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, ecrNode1)},
        {QMDDEdge(1.0, ecrNode2), QMDDEdge(.0, nullptr)}
    })));
}

QMDDGate gate::fSim(double theta, double phi) {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, theta]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(cos(theta), nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4, phi, theta]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(exp(i * phi) * mathUtils::sec(theta), nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> fSimNode1 = future1.get();
    shared_ptr<QMDDNode> fSimNode2 = future2.get();
    shared_ptr<QMDDNode> fSimNode3 = future3.get();
    shared_ptr<QMDDNode> fSimNode4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, fSimNode1), QMDDEdge(-i * sin(theta), fSimNode2)},
        {QMDDEdge(-i * sin(theta), fSimNode3), QMDDEdge(cos(theta), fSimNode4)}
    })));
}

QMDDGate gate::G(double theta) {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, theta]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(cos(theta), nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4, theta]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0 * mathUtils::sec(theta), nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> gNode1 = future1.get();
    shared_ptr<QMDDNode> gNode2 = future2.get();
    shared_ptr<QMDDNode> gNode3 = future3.get();
    shared_ptr<QMDDNode> gNode4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, gNode1), QMDDEdge(-sin(theta), gNode2)},
        {QMDDEdge(sin(theta), gNode3), QMDDEdge(cos(theta), gNode4)}
    })));
}

QMDDGate gate::M() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(-i, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(i, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> mNode1 = future1.get();
    shared_ptr<QMDDNode> mNode2 = future2.get();
    shared_ptr<QMDDNode> mNode3 = future3.get();
    shared_ptr<QMDDNode> mNode4 = future4.get();

    return QMDDGate(QMDDEdge(1.0 / sqrt(2.0), make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, mNode1), QMDDEdge(i, mNode2)},
        {QMDDEdge(1.0, mNode3), QMDDEdge(i, mNode4)}
    })));
}

QMDDGate gate::syc() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> sycNode1 = future1.get();
    shared_ptr<QMDDNode> sycNode2 = future2.get();
    shared_ptr<QMDDNode> sycNode3 = future3.get();
    shared_ptr<QMDDNode> sycNode4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, sycNode1), QMDDEdge(-i, sycNode2)},
        {QMDDEdge(-i, sycNode3), QMDDEdge(exp(-i * M_PI / 6.0), sycNode4)}
    })));
}

QMDDGate gate::CZS(double theta, double phi, double gamma) {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();

    boost::fibers::fiber([&promise1, theta, gamma]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(-exp(i * gamma) * std::pow(sin(theta / 2.0), 2) + std::pow(cos(theta / 2.0), 2), nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4, theta, gamma]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(-exp(i * gamma) / (-exp(i * gamma) * std::pow(cos(theta / 2.0), 2) + std::pow(sin(theta / 2.0), 2)), nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();

    shared_ptr<QMDDNode> czsNode1 = future1.get();
    shared_ptr<QMDDNode> czsNode2 = future2.get();
    shared_ptr<QMDDNode> czsNode3 = future3.get();
    shared_ptr<QMDDNode> czsNode4 = future4.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, czsNode1), QMDDEdge((1.0 + exp(i * gamma)) / 2.0 * exp(-i * phi) * sin(theta), czsNode2)},
        {QMDDEdge((1.0 + exp(i * gamma)) / 2.0 * exp(i * phi) * sin(theta), czsNode3), QMDDEdge(-exp(i * gamma) * std::pow(cos(theta / 2.0), 2) + std::pow(sin(theta / 2.0), 2), czsNode4)}
    })));
}

QMDDGate gate::D(double theta) {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
        }));
    }).detach();

    boost::fibers::fiber([&promise2, theta]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(-i * tan(theta), nullptr)},
            {QMDDEdge(-i * tan(theta), nullptr), QMDDEdge(1.0, nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();

    shared_ptr<QMDDNode> dNode1 = future1.get();
    shared_ptr<QMDDNode> dNode2 = future2.get();

    shared_ptr<QMDDNode> dNode3 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(i * cos(theta), dNode2)}
    });

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, dNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, dNode3)}
    })));
}

QMDDGate gate::RCCX() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::Z().getStartNode())), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))}
        }));
    }).detach();

    future1.wait();
    future2.wait();

    shared_ptr<QMDDNode> rccxNode1 = future1.get();
    shared_ptr<QMDDNode> rccxNode2 = future2.get();

    return QMDDGate(QMDDEdge(1.0, make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
        {QMDDEdge(1.0, rccxNode1), QMDDEdge(.0, nullptr)},
        {QMDDEdge(.0, nullptr), QMDDEdge(1.0, rccxNode2)}
    })));
}

QMDDGate gate::PG() {

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::X().getStartNode()))},
            {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();

    shared_ptr<QMDDNode> pgNode1 = future1.get();
    shared_ptr<QMDDNode> pgNode2 = future2.get();

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

    boost::fibers::use_scheduling_algorithm<CustomScheduler>();

    boost::fibers::promise<shared_ptr<QMDDNode>> promise1, promise2, promise3, promise4, promise5;
    boost::fibers::future<shared_ptr<QMDDNode>> future1 = promise1.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future2 = promise2.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future3 = promise3.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future4 = promise4.get_future();
    boost::fibers::future<shared_ptr<QMDDNode>> future5 = promise5.get_future();

    boost::fibers::fiber([&promise1]() {
        promise1.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode())), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, make_shared<QMDDNode>(*gate::I().getStartNode()))}
        }));
    }).detach();

    boost::fibers::fiber([&promise2]() {
        promise2.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise3]() {
        promise3.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(1.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise4]() {
        promise4.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)}
        }));
    }).detach();

    boost::fibers::fiber([&promise5]() {
        promise5.set_value(make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
            {QMDDEdge(.0, nullptr), QMDDEdge(.0, nullptr)},
            {QMDDEdge(.0, nullptr), QMDDEdge(1.0, nullptr)}
        }));
    }).detach();

    future1.wait();
    future2.wait();
    future3.wait();
    future4.wait();
    future5.wait();

    shared_ptr<QMDDNode> fFredkinNode1 = future1.get();
    shared_ptr<QMDDNode> fFredkinNode2 = future2.get();
    shared_ptr<QMDDNode> fFredkinNode3 = future3.get();
    shared_ptr<QMDDNode> fFredkinNode4 = future4.get();
    shared_ptr<QMDDNode> fFredkinNode5 = future5.get();


    shared_ptr<QMDDNode> fFredkinNode6 = make_shared<QMDDNode>(vector<vector<QMDDEdge>>{
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