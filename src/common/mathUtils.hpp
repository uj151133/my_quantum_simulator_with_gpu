#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include <boost/fiber/all.hpp>
#include <boost/fiber/algo/work_stealing.hpp>
#include <boost/fiber/mutex.hpp>
#include <cmath>
#include <limits>
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
#include "parameter.hpp"
#include "calculation.hpp"
#include "../modules/threadPool.hpp"
#include "../common/operationCacheClient.hpp"
#include "../common/bigBang.hpp"
#include "../modules/backend.hpp"
#include "../modules/Heian.h"

using namespace std;
// using namespace Eigen;

namespace mathUtils {
    QMDDEdge mul(const QMDDEdge& e0, const QMDDEdge& e1, bool onFiber = false, int depth = 0);
    // QMDDEdge mulForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge add(const QMDDEdge& e0, const QMDDEdge& e1, int depth = 0);
    // QMDDEdge addForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1);
    QMDDEdge kron(const QMDDEdge& e0, const QMDDEdge& e1, int depth = 0);
    // QMDDEdge kron(const QMDDEdge& e0, const QMDDEdge& e1);
    // QMDDEdge kronForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1);

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

    bool isMultiplePI(double theta, double eps = 1e-10);

    bool isZERO(const complex<double>& z);

    complex<double> normalize(vector<vector<QMDDEdge>>& e, bool& allWeightsAreZero);

    QMDDEdge beAuthentic(const QMDDEdge& e, int end);

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

    template <class T, size_t N>
    double mean(const array<T, N>& a) {
        static_assert(N > 0, "mean: array must be non-empty");
        double sum = accumulate(a.begin(), a.end(), 0.0, [](double acc, const T& v) {
            return acc + static_cast<double>(v);
        });
        return sum / static_cast<double>(N);
    }

    template <class T, size_t N>
    double min(const array<T, N>& a) {
        static_assert(N > 0, "min: array must be non-empty");
        const T& m = *min_element(a.begin(), a.end());
        return static_cast<double>(m);
    }
}

#endif // MATH_UTILS_HPP
