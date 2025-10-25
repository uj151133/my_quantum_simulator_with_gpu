#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include <boost/fiber/all.hpp>
#include <boost/fiber/algo/work_stealing.hpp>
#include <boost/fiber/mutex.hpp>
#include <cmath>
#include <random>
#include <numeric>
#include <queue>
#include <stack>
#include <functional>
#include <thread>
// #include <boost/asio/io_service.hpp>
#include <boost/bind/bind.hpp>
#include <boost/thread/thread.hpp>
#include "../models/qmdd.hpp"
#include "../models/uniqueTable.hpp"
#include "config.hpp"
#include "calculation.hpp"
#include "../modules/threadPool.hpp"
#include <tbb/task_group.h>
#include "constant.hpp"
#include "../common/operationCacheClient.hpp"


using namespace std;
// using namespace Eigen;

namespace mathUtils {
    QMDDEdge mul(const QMDDEdge& e0, const QMDDEdge& e1, bool parallelism = false, bool concurrency = false);
    QMDDEdge mulForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge add(const QMDDEdge& e0, const QMDDEdge& e1, bool parallelism = false, bool concurrency = false);
    QMDDEdge addForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge kron(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge kronForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1);

    QMDDEdge dyad(const QMDDEdge& e0, const QMDDEdge& e1);

    double csc(double theta);
    complex<double> csc(complex<double> theta);
    double sec(double theta);
    complex<double> sec(complex<double> theta);
    double cot(double theta);
    complex<double> cot(complex<double> theta);

    double sumOfSquares(const vector<complex<double>>& vec);

    vector<int> createRange(int start, int end);
    int findCoprimeBelow(int N);

    template <class T, size_t N>
    double median(const array<T, N>& a) {
        static_assert(N > 0, "median: array must be non-empty");
        auto b = a;
        const size_t mid1 = (N - 1) / 2;
        const size_t mid2 = N / 2;
        nth_element(b.begin(), b.begin() + mid1, b.end());
        const T x = b[mid1];
        nth_element(b.begin(), b.begin() + mid2, b.end());
        const T y = b[mid2];
        if constexpr (N % 2 == 1) {
            return static_cast<double>(x);
        } else {
            return 0.5 * (static_cast<double>(x) + static_cast<double>(y));
        }
    }
}

#endif // MATH_UTILS_HPP
