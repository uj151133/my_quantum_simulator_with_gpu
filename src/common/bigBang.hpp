#ifndef BIGBANG_HPP
#define BIGBANG_HPP

#include <complex>

#include "parameter.hpp"
#include "../modules/threadPool.hpp"
#include "../models/uniqueTable.hpp"
#include "../models/gate.hpp"
#include "../models/state.hpp"
#include "../common/operationCacheClient.hpp"
#include "../models/qmdd.hpp"

using namespace std;

extern complex<double> i;
extern QMDDEdge edgeZero;
extern QMDDEdge edgeOne;

extern QMDDEdge identityEdge;
extern QMDDEdge braketZero;
extern QMDDEdge braketOne;

void birth();

#endif