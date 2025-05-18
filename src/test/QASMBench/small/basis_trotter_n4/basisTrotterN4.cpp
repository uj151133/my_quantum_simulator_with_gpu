#include "basisTrotterN4.hpp"

void basisTrotterN4() {

    QuantumCircuit q(4);
    vector<int> c(4);


    q.addZ(0);
    q.addZ(1);

    q.addZ(2);
    q.addZ(3);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);


    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);


    q.addCX(0, 1);
    q.addH(0);
    q.addCX(1, 0);
    q.addRz(0, M_PI * 0.0406530731);
    q.addCX(1, 0);
    q.addRz(0, M_PI * -0.0406530731);
    q.addH(0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(3, M_PI * -0.25);


    q.addCX(2, 3);
    q.addH(2);
    q.addCX(3, 2);
    q.addRz(2, M_PI * -0.0406530731);
    q.addCX(3, 2);
    q.addRz(2, M_PI * 0.0406530731);
    q.addH(2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * -0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(0, M_PI * 0.1123177385);
    q.addRz(1, M_PI * 0.25);

    q.addRz(2, M_PI * -0.25);


    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * 0.1123177385);
    q.addRz(3, M_PI * 0.0564909955);

    q.addRz(2, M_PI * 0.0564909955);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);


    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);


    q.addCX(0, 1);
    q.addH(0);
    q.addCX(1, 0);
    q.addRz(0, M_PI * -0.0255147541);
    q.addCX(1, 0);
    q.addRz(0, M_PI * 0.0255147541);
    q.addH(0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(3, M_PI * -0.25);


    q.addCX(2, 3);
    q.addH(2);
    q.addCX(3, 2);
    q.addRz(2, M_PI * 0.0255147541);
    q.addCX(3, 2);
    q.addRz(2, M_PI * -0.0255147541);
    q.addH(2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * -0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);


    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addU3(0, M_PI * 0.5, 0, M_PI * 0.5);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0);


    q.addCX(0, 1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * 0.5);
    q.addCX(0, 1);
    q.addU3(0, M_PI * 0.5, M_PI * 0.4758602045, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.9758602045, 0);
    q.addSWAP(0, 1);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.75);
    q.addU3(3, M_PI * 0.5, M_PI * 1.0, M_PI * 1.25);
    q.addCX(2, 3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * 0.5);

    q.addCX(2, 3);


    q.addU3(2, M_PI * 0.5, M_PI * 1.2389215436, M_PI * 1.0);
    q.addU3(3, M_PI * 0.5, M_PI * 1.7389215436, 0);
    q.addSWAP(2, 3);
    q.addU3(1, M_PI * 0.5, 0, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(1, 2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * 0.5);
    q.addCX(1, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4836466618, 0);
    q.addSWAP(1, 2);

    q.addU3(0, M_PI * 0.5, 0, 0);


    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(0, 1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * 0.5);
    q.addCX(0, 1);
    q.addU3(0, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.4836466618, 0);
    q.addU3(2, M_PI * 0.5, 0, 0);
    q.addU3(3, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(2, 3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * 0.5);

    q.addCX(2, 3);


    q.addU3(2, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(3, M_PI * 0.5, M_PI * 1.4836466618, 0);
    q.addSWAP(0, 1);
    q.addSWAP(2, 3);
    q.addU3(1, M_PI * 0.5, 0, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(1, 2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * 0.5);
    q.addCX(1, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4836466618, 0);


    q.addRz(3, M_PI * -0.0241397955);
    q.addRz(0, M_PI * -0.0110784564);
    q.addSWAP(1, 2);
    q.addRz(2, M_PI * -0.0241397955);
    q.addRz(1, M_PI * -0.0110784564);
    q.addZ(2);
    q.addZ(1);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(2, 1);
    q.addH(2);
    q.addCX(1, 2);

    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);


    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(3, 2);
    q.addH(3);
    q.addCX(2, 3);
    q.addRz(3, M_PI * -0.4750315453);
    q.addCX(2, 3);

    q.addRz(3, M_PI * 0.4750315453);
    q.addH(3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(0, M_PI * -0.25);


    q.addCX(1, 0);
    q.addH(1);
    q.addCX(0, 1);
    q.addRz(1, M_PI * 0.4750315453);
    q.addCX(0, 1);
    q.addRz(1, M_PI * -0.4750315453);
    q.addH(1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * -0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);


    q.addCX(2, 1);
    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addU3(3, M_PI * 0.5, 0, M_PI * 1.5);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0);


    q.addCX(3, 2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * 0.5);
    q.addCX(3, 2);
    q.addU3(3, M_PI * 0.5, M_PI * 1.4931729076, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.9931729076, 0);
    q.addSWAP(3, 2);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.4961253835);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0, M_PI * 1.9961253835);
    q.addCX(1, 0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * 0.5);


    q.addCX(1, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.5007105964, M_PI * 1.0);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0007105964, 0);
    q.addSWAP(1, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(2, 1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * 0.5);
    q.addCX(2, 1);
    q.addU3(2, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.4225955389, 0);


    q.addSWAP(2, 1);
    q.addU3(3, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(3, 2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * 0.5);
    q.addCX(3, 2);
    q.addU3(3, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4225955389, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(1, 0);

    q.addCX(0, 1);


    q.addRz(0, M_PI * 0.5);
    q.addCX(1, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(0, M_PI * 0.5, M_PI * 1.4225955389, 0);
    q.addSWAP(3, 2);
    q.addSWAP(1, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(2, 1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * 0.5);
    q.addCX(2, 1);

    q.addU3(2, M_PI * 0.5, M_PI * 1.9225955389, 0);


    q.addU3(1, M_PI * 0.5, M_PI * 1.4225955389, 0);
    q.addRz(0, M_PI * -0.0068270924);
    q.addRz(3, M_PI * -0.0031640201);
    q.addSWAP(2, 1);
    q.addZ(0);
    q.addZ(3);
    q.addRz(1, M_PI * -0.0068270924);
    q.addRz(2, M_PI * -0.0031640201);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);
    q.addH(1);

    q.addCX(2, 1);


    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(0, 1);
    q.addH(0);
    q.addCX(1, 0);


    q.addRz(0, M_PI * -0.2508765254);
    q.addCX(1, 0);
    q.addRz(0, M_PI * 0.2508765254);
    q.addH(0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(3, M_PI * -0.25);
    q.addCX(2, 3);
    q.addH(2);
    q.addCX(3, 2);

    q.addRz(2, M_PI * 0.2508765254);
    q.addCX(3, 2);


    q.addRz(2, M_PI * -0.2508765254);
    q.addH(2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * -0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);

    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addU3(0, M_PI * 0.5, 0, M_PI * 1.5001274262);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0001274262);


    q.addCX(0, 1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * 0.5);
    q.addCX(0, 1);
    q.addU3(0, M_PI * 0.5, M_PI * 1.4996406983, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.9996406983, 0);
    q.addSWAP(0, 1);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.4998373235);
    q.addU3(3, M_PI * 0.5, 0, M_PI * 1.9998373235);
    q.addCX(2, 3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * 0.5);


    q.addCX(2, 3);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4999562012, 0);
    q.addU3(3, M_PI * 0.5, M_PI * 0.9999562012, M_PI * 1.0);
    q.addSWAP(2, 3);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(1, 2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * 0.5);
    q.addCX(1, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);


    q.addSWAP(1, 2);
    q.addU3(0, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(0, 1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * 0.5);
    q.addCX(0, 1);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(3, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(2, 3);


    q.addCX(3, 2);
    q.addRz(3, M_PI * 0.5);
    q.addCX(2, 3);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(3, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);
    q.addSWAP(0, 1);
    q.addSWAP(2, 3);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(1, 2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * 0.5);


    q.addCX(1, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);
    q.addRz(3, M_PI * -0.0002318755);
    q.addRz(0, M_PI * -0.0002064753);
    q.addSWAP(1, 2);
    q.addZ(3);
    q.addZ(0);
    q.addRz(2, M_PI * -0.0002318755);
    q.addRz(1, M_PI * -0.0002064753);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);

    q.addCX(2, 1);


    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(3, 2);

    q.addH(3);


    q.addCX(2, 3);
    q.addRz(3, M_PI * -0.2079241021);
    q.addCX(2, 3);
    q.addRz(3, M_PI * 0.2079241021);
    q.addH(3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(0, M_PI * -0.25);
    q.addCX(1, 0);
    q.addH(1);

    q.addCX(0, 1);


    q.addRz(1, M_PI * 0.2079241021);
    q.addCX(0, 1);
    q.addRz(1, M_PI * -0.2079241021);
    q.addH(1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * -0.25);
    q.addRz(0, M_PI * 0.25);
    q.addZ(3);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(2, 1);
    q.addH(2);


    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addZ(2);
    q.addZ(0);
    q.addZ(1);
    q.addRz(2, M_PI * 0.25);

    q.addRz(1, M_PI * -0.25);
    q.addCX(2, 1);


    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(3, 2);

    q.addH(3);
    q.addCX(2, 3);
    q.addRz(3, M_PI * 0.0406530731);
    q.addCX(2, 3);
    q.addRz(3, M_PI * -0.0406530731);
    q.addH(3);
    q.addCX(3, 2);


    q.addRz(3, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(0, M_PI * -0.25);
    q.addCX(1, 0);
    q.addH(1);
    q.addCX(0, 1);
    q.addRz(1, M_PI * -0.0406530731);
    q.addCX(0, 1);
    q.addRz(1, M_PI * 0.0406530731);
    q.addH(1);
    q.addCX(1, 0);


    q.addRz(1, M_PI * -0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(3, M_PI * 0.1123177385);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(2, 1);
    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);


    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * 0.1123177385);
    q.addRz(0, M_PI * 0.0564909955);
    q.addRz(1, M_PI * 0.0564909955);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(2, 1);
    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);

    q.addCX(1, 2);


    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(3, 2);
    q.addH(3);
    q.addCX(2, 3);
    q.addRz(3, M_PI * -0.0255147541);
    q.addCX(2, 3);

    q.addRz(3, M_PI * 0.0255147541);
    q.addH(3);
    q.addCX(3, 2);


    q.addRz(3, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(0, M_PI * -0.25);
    q.addCX(1, 0);
    q.addH(1);
    q.addCX(0, 1);
    q.addRz(1, M_PI * 0.0255147541);
    q.addCX(0, 1);
    q.addRz(1, M_PI * -0.0255147541);
    q.addH(1);
    q.addCX(1, 0);


    q.addRz(1, M_PI * -0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(2, 1);
    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);


    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addU3(3, M_PI * 0.5, 0, M_PI * 0.5);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0);
    q.addCX(3, 2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * 0.5);
    q.addCX(3, 2);
    q.addU3(3, M_PI * 0.5, M_PI * 0.4758602045, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.9758602045, 0);
    q.addSWAP(3, 2);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.75);

    q.addU3(0, M_PI * 0.5, M_PI * 1.0, M_PI * 1.25);


    q.addCX(1, 0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * 0.5);
    q.addCX(1, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.2389215436, M_PI * 1.0);
    q.addU3(0, M_PI * 0.5, M_PI * 1.7389215436, 0);
    q.addSWAP(1, 0);
    q.addU3(2, M_PI * 0.5, 0, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(2, 1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * 0.5);

    q.addCX(2, 1);
    q.addU3(2, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.4836466618, 0);


    q.addSWAP(2, 1);
    q.addU3(3, M_PI * 0.5, 0, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(3, 2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * 0.5);
    q.addCX(3, 2);
    q.addU3(3, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4836466618, 0);
    q.addU3(1, M_PI * 0.5, 0, 0);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(1, 0);


    q.addCX(0, 1);
    q.addRz(0, M_PI * 0.5);
    q.addCX(1, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(0, M_PI * 0.5, M_PI * 1.4836466618, 0);
    q.addSWAP(3, 2);
    q.addSWAP(1, 0);
    q.addU3(2, M_PI * 0.5, 0, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(2, 1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * 0.5);


    q.addCX(2, 1);
    q.addU3(2, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.4836466618, 0);
    q.addRz(0, M_PI * -0.0241397955);
    q.addRz(3, M_PI * -0.0110784564);
    q.addSWAP(2, 1);
    q.addRz(1, M_PI * -0.0241397955);
    q.addRz(2, M_PI * -0.0110784564);
    q.addZ(1);
    q.addZ(2);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);


    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);


    q.addCX(0, 1);
    q.addH(0);
    q.addCX(1, 0);
    q.addRz(0, M_PI * -0.4750315453);
    q.addCX(1, 0);
    q.addRz(0, M_PI * 0.4750315453);
    q.addH(0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(3, M_PI * -0.25);

    q.addCX(2, 3);


    q.addH(2);
    q.addCX(3, 2);
    q.addRz(2, M_PI * 0.4750315453);
    q.addCX(3, 2);
    q.addRz(2, M_PI * -0.4750315453);
    q.addH(2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * -0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);

    q.addH(1);


    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addU3(0, M_PI * 0.5, 0, M_PI * 1.5);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0);
    q.addCX(0, 1);
    q.addCX(1, 0);

    q.addRz(1, M_PI * 0.5);


    q.addCX(0, 1);
    q.addU3(0, M_PI * 0.5, M_PI * 1.4931729076, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.9931729076, 0);
    q.addSWAP(0, 1);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.4961253835);
    q.addU3(3, M_PI * 0.5, M_PI * 1.0, M_PI * 1.9961253835);
    q.addCX(2, 3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * 0.5);
    q.addCX(2, 3);
    q.addU3(2, M_PI * 0.5, M_PI * 1.5007105964, M_PI * 1.0);
    q.addU3(3, M_PI * 0.5, M_PI * 1.0007105964, 0);


    q.addSWAP(2, 3);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(1, 2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * 0.5);
    q.addCX(1, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4225955389, 0);
    q.addSWAP(1, 2);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);

    q.addCX(0, 1);
    q.addCX(1, 0);


    q.addRz(1, M_PI * 0.5);
    q.addCX(0, 1);
    q.addU3(0, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.4225955389, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(3, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(2, 3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * 0.5);
    q.addCX(2, 3);
    q.addU3(2, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(3, M_PI * 0.5, M_PI * 1.4225955389, 0);

    q.addSWAP(0, 1);
    q.addSWAP(2, 3);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(1, 2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * 0.5);


    q.addCX(1, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4225955389, 0);
    q.addRz(3, M_PI * -0.0068270924);
    q.addRz(0, M_PI * -0.0031640201);
    q.addSWAP(1, 2);
    q.addZ(3);
    q.addZ(0);
    q.addRz(2, M_PI * -0.0068270924);
    q.addRz(1, M_PI * -0.0031640201);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);


    q.addCX(2, 1);
    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);


    q.addCX(3, 2);
    q.addH(3);
    q.addCX(2, 3);
    q.addRz(3, M_PI * -0.2508765254);
    q.addCX(2, 3);
    q.addRz(3, M_PI * 0.2508765254);
    q.addH(3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(0, M_PI * -0.25);


    q.addCX(1, 0);
    q.addH(1);
    q.addCX(0, 1);
    q.addRz(1, M_PI * 0.2508765254);
    q.addCX(0, 1);
    q.addRz(1, M_PI * -0.2508765254);
    q.addH(1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * -0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);


    q.addCX(2, 1);
    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addU3(3, M_PI * 0.5, 0, M_PI * 1.5001274262);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0001274262);

    q.addCX(3, 2);


    q.addCX(2, 3);
    q.addRz(2, M_PI * 0.5);
    q.addCX(3, 2);
    q.addU3(3, M_PI * 0.5, M_PI * 1.4996406983, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.9996406983, 0);
    q.addSWAP(3, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.4998373235);
    q.addU3(0, M_PI * 0.5, 0, M_PI * 1.9998373235);
    q.addCX(1, 0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * 0.5);
    q.addCX(1, 0);

    q.addU3(1, M_PI * 0.5, M_PI * 1.4999562012, 0);


    q.addU3(0, M_PI * 0.5, M_PI * 0.9999562012, M_PI * 1.0);
    q.addSWAP(1, 0);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(2, 1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * 0.5);
    q.addCX(2, 1);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);
    q.addSWAP(2, 1);
    q.addU3(3, M_PI * 0.5, 0, M_PI * 1.9993457511);

    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.4993457511);


    q.addCX(3, 2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * 0.5);
    q.addCX(3, 2);
    q.addU3(3, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(0, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(1, 0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * 0.5);
    q.addCX(1, 0);


    q.addU3(1, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(0, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);
    q.addSWAP(3, 2);
    q.addSWAP(1, 0);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(2, 1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * 0.5);
    q.addCX(2, 1);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);

    q.addRz(0, M_PI * -0.0002318755);
    q.addRz(3, M_PI * -0.0002064753);


    q.addSWAP(2, 1);
    q.addZ(0);
    q.addZ(3);
    q.addRz(1, M_PI * -0.0002318755);
    q.addRz(2, M_PI * -0.0002064753);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);

    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);


    q.addCX(0, 1);
    q.addH(0);
    q.addCX(1, 0);
    q.addRz(0, M_PI * -0.2079241021);
    q.addCX(1, 0);
    q.addRz(0, M_PI * 0.2079241021);
    q.addH(0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(3, M_PI * -0.25);


    q.addCX(2, 3);
    q.addH(2);
    q.addCX(3, 2);
    q.addRz(2, M_PI * 0.2079241021);
    q.addCX(3, 2);
    q.addRz(2, M_PI * -0.2079241021);
    q.addH(2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * -0.25);
    q.addRz(3, M_PI * 0.25);
    q.addZ(0);
    q.addRz(1, M_PI * 0.25);


    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addZ(1);


    q.addZ(3);
    q.addZ(2);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);


    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(0, 1);
    q.addH(0);
    q.addCX(1, 0);
    q.addRz(0, M_PI * 0.0406530731);
    q.addCX(1, 0);
    q.addRz(0, M_PI * -0.0406530731);
    q.addH(0);
    q.addCX(0, 1);

    q.addRz(0, M_PI * -0.25);


    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(3, M_PI * -0.25);
    q.addCX(2, 3);
    q.addH(2);
    q.addCX(3, 2);
    q.addRz(2, M_PI * -0.0406530731);
    q.addCX(3, 2);
    q.addRz(2, M_PI * 0.0406530731);
    q.addH(2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * -0.25);

    q.addRz(3, M_PI * 0.25);


    q.addRz(0, M_PI * 0.1123177385);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);

    q.addRz(2, M_PI * 0.25);


    q.addRz(1, M_PI * 0.1123177385);
    q.addRz(3, M_PI * 0.0564909955);
    q.addRz(2, M_PI * 0.0564909955);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);


    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(0, 1);
    q.addH(0);
    q.addCX(1, 0);
    q.addRz(0, M_PI * -0.0255147541);
    q.addCX(1, 0);
    q.addRz(0, M_PI * 0.0255147541);
    q.addH(0);

    q.addCX(0, 1);
    q.addRz(0, M_PI * -0.25);


    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(3, M_PI * -0.25);
    q.addCX(2, 3);
    q.addH(2);
    q.addCX(3, 2);
    q.addRz(2, M_PI * 0.0255147541);
    q.addCX(3, 2);
    q.addRz(2, M_PI * -0.0255147541);
    q.addH(2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * -0.25);

    q.addRz(3, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);


    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addU3(0, M_PI * 0.5, 0, M_PI * 0.5);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0);
    q.addCX(0, 1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * 0.5);
    q.addCX(0, 1);


    q.addU3(0, M_PI * 0.5, M_PI * 0.4758602045, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.9758602045, 0);
    q.addSWAP(0, 1);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.75);
    q.addU3(3, M_PI * 0.5, M_PI * 1.0, M_PI * 1.25);
    q.addCX(2, 3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * 0.5);
    q.addCX(2, 3);
    q.addU3(2, M_PI * 0.5, M_PI * 1.2389215436, M_PI * 1.0);
    q.addU3(3, M_PI * 0.5, M_PI * 1.7389215436, 0);
    q.addSWAP(2, 3);


    q.addU3(1, M_PI * 0.5, 0, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(1, 2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * 0.5);
    q.addCX(1, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4836466618, 0);
    q.addSWAP(1, 2);
    q.addU3(0, M_PI * 0.5, 0, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(0, 1);

    q.addCX(1, 0);


    q.addRz(1, M_PI * 0.5);
    q.addCX(0, 1);
    q.addU3(0, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.4836466618, 0);
    q.addU3(2, M_PI * 0.5, 0, 0);
    q.addU3(3, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(2, 3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * 0.5);
    q.addCX(2, 3);
    q.addU3(2, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(3, M_PI * 0.5, M_PI * 1.4836466618, 0);

    q.addSWAP(0, 1);
    q.addSWAP(2, 3);
    q.addU3(1, M_PI * 0.5, 0, 0);


    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5);
    q.addCX(1, 2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * 0.5);
    q.addCX(1, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 0.9836466618, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4836466618, 0);
    q.addRz(3, M_PI * -0.0241397955);
    q.addRz(0, M_PI * -0.0110784564);
    q.addSWAP(1, 2);
    q.addRz(2, M_PI * -0.0241397955);
    q.addRz(1, M_PI * -0.0110784564);


    q.addZ(2);
    q.addZ(1);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(2, 1);
    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);


    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(3, 2);
    q.addH(3);
    q.addCX(2, 3);
    q.addRz(3, M_PI * -0.4750315453);
    q.addCX(2, 3);
    q.addRz(3, M_PI * 0.4750315453);
    q.addH(3);
    q.addCX(3, 2);

    q.addRz(3, M_PI * -0.25);


    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(0, M_PI * -0.25);
    q.addCX(1, 0);
    q.addH(1);
    q.addCX(0, 1);
    q.addRz(1, M_PI * 0.4750315453);
    q.addCX(0, 1);
    q.addRz(1, M_PI * -0.4750315453);
    q.addH(1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * -0.25);

    q.addRz(0, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);


    q.addCX(2, 1);
    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addU3(3, M_PI * 0.5, 0, M_PI * 1.5);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0);


    q.addCX(3, 2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * 0.5);
    q.addCX(3, 2);
    q.addU3(3, M_PI * 0.5, M_PI * 1.4931729076, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.9931729076, 0);
    q.addSWAP(3, 2);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.4961253835);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0, M_PI * 1.9961253835);
    q.addCX(1, 0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * 0.5);


    q.addCX(1, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.5007105964, M_PI * 1.0);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0007105964, 0);
    q.addSWAP(1, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(2, 1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * 0.5);
    q.addCX(2, 1);
    q.addU3(2, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.4225955389, 0);


    q.addSWAP(2, 1);
    q.addU3(3, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(3, 2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * 0.5);
    q.addCX(3, 2);
    q.addU3(3, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4225955389, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(1, 0);


    q.addCX(0, 1);
    q.addRz(0, M_PI * 0.5);
    q.addCX(1, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(0, M_PI * 0.5, M_PI * 1.4225955389, 0);
    q.addSWAP(3, 2);
    q.addSWAP(1, 0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0820521548);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.5820521548);
    q.addCX(2, 1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * 0.5);

    q.addCX(2, 1);


    q.addU3(2, M_PI * 0.5, M_PI * 1.9225955389, 0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.4225955389, 0);
    q.addRz(0, M_PI * -0.0068270924);
    q.addRz(3, M_PI * -0.0031640201);
    q.addSWAP(2, 1);
    q.addZ(0);
    q.addZ(3);
    q.addRz(1, M_PI * -0.0068270924);
    q.addRz(2, M_PI * -0.0031640201);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);

    q.addH(1);


    q.addCX(2, 1);
    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);
    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(0, 1);
    q.addH(0);

    q.addCX(1, 0);


    q.addRz(0, M_PI * -0.2508765254);
    q.addCX(1, 0);
    q.addRz(0, M_PI * 0.2508765254);
    q.addH(0);
    q.addCX(0, 1);
    q.addRz(0, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(3, M_PI * -0.25);
    q.addCX(2, 3);
    q.addH(2);
    q.addCX(3, 2);


    q.addRz(2, M_PI * 0.2508765254);
    q.addCX(3, 2);
    q.addRz(2, M_PI * -0.2508765254);
    q.addH(2);
    q.addCX(2, 3);
    q.addRz(2, M_PI * -0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);
    q.addCX(1, 2);
    q.addH(1);
    q.addCX(2, 1);

    q.addRz(1, M_PI * -0.5);
    q.addCX(2, 1);


    q.addRz(1, M_PI * 0.5);
    q.addH(1);
    q.addCX(1, 2);
    q.addRz(1, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addU3(0, M_PI * 0.5, 0, M_PI * 1.5001274262);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0, M_PI * 1.0001274262);
    q.addCX(0, 1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * 0.5);
    q.addCX(0, 1);
    q.addU3(0, M_PI * 0.5, M_PI * 1.4996406983, M_PI * 1.0);

    q.addU3(1, M_PI * 0.5, M_PI * 1.9996406983, 0);
    q.addSWAP(0, 1);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0, M_PI * 1.4998373235);
    q.addU3(3, M_PI * 0.5, 0, M_PI * 1.9998373235);
    q.addCX(2, 3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * 0.5);


    q.addCX(2, 3);
    q.addU3(2, M_PI * 0.5, M_PI * 1.4999562012, 0);
    q.addU3(3, M_PI * 0.5, M_PI * 0.9999562012, M_PI * 1.0);
    q.addSWAP(2, 3);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(1, 2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * 0.5);
    q.addCX(1, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);


    q.addSWAP(1, 2);
    q.addU3(0, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(0, 1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * 0.5);
    q.addCX(0, 1);
    q.addU3(0, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(1, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(3, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(2, 3);


    q.addCX(3, 2);
    q.addRz(3, M_PI * 0.5);
    q.addCX(2, 3);
    q.addU3(2, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(3, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);
    q.addSWAP(0, 1);
    q.addSWAP(2, 3);
    q.addU3(1, M_PI * 0.5, 0, M_PI * 1.9993457511);
    q.addU3(2, M_PI * 0.5, 0, M_PI * 1.4993457511);
    q.addCX(1, 2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * 0.5);


    q.addCX(1, 2);
    q.addU3(1, M_PI * 0.5, M_PI * 1.0008730561, M_PI * 1.0);
    q.addU3(2, M_PI * 0.5, M_PI * 1.5008730561, M_PI * 1.0);
    q.addRz(3, M_PI * -0.0002318755);
    q.addRz(0, M_PI * -0.0002064753);
    q.addSWAP(1, 2);
    q.addZ(3);
    q.addZ(0);
    q.addRz(2, M_PI * -0.0002318755);
    q.addRz(1, M_PI * -0.0002064753);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);


    q.addCX(2, 1);
    q.addH(2);
    q.addCX(1, 2);
    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(3, M_PI * 0.25);
    q.addRz(2, M_PI * -0.25);

    q.addCX(3, 2);


    q.addH(3);
    q.addCX(2, 3);
    q.addRz(3, M_PI * -0.2079241021);
    q.addCX(2, 3);
    q.addRz(3, M_PI * 0.2079241021);
    q.addH(3);
    q.addCX(3, 2);
    q.addRz(3, M_PI * -0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * 0.25);
    q.addRz(0, M_PI * -0.25);
    q.addCX(1, 0);

    q.addH(1);


    q.addCX(0, 1);
    q.addRz(1, M_PI * 0.2079241021);
    q.addCX(0, 1);
    q.addRz(1, M_PI * -0.2079241021);
    q.addH(1);
    q.addCX(1, 0);
    q.addRz(1, M_PI * -0.25);
    q.addRz(0, M_PI * 0.25);
    q.addRz(2, M_PI * 0.25);
    q.addRz(1, M_PI * -0.25);
    q.addCX(2, 1);
    q.addH(2);

    q.addCX(1, 2);


    q.addRz(2, M_PI * -0.5);
    q.addCX(1, 2);
    q.addRz(2, M_PI * 0.5);
    q.addH(2);
    q.addCX(2, 1);
    q.addRz(2, M_PI * -0.25);
    q.addRz(1, M_PI * 0.25);
    q.addSWAP(3, 2);
    q.addSWAP(1, 0);
    q.addSWAP(2, 1);
    q.addSWAP(3, 2);
    q.addSWAP(1, 0);


    q.addSWAP(2, 1);
    c[0] = q.measure(0);
    c[1] = q.measure(1);
    c[2] = q.measure(2);
    c[3] = q.measure(3);
    
    
    
    
    
    
    

    
    


    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    

    


    
    
    
    
    
    
    
    
    
    
    
    

    


    
    
    
    
    
    
    
    
    
    
    
    

    


    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    

    
    


    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    

    
    
    
    

    return;
}
