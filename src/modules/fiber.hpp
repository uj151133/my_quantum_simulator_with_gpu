#ifndef FIBER_HPP
#define FIBER_HPP

#include <boost/fiber/all.hpp>
#include <boost/fiber/mutex.hpp>
#include <boost/fiber/condition_variable.hpp>
#include <vector>
#include <queue>
#include <thread>
#include <functional>

class FiberPool {
private:
    std::vector<boost::fibers::fiber> workers;
    std::queue<std::function<void()>> tasks;
    boost::fibers::mutex queue_mutex;
    boost::fibers::condition_variable condition;
    bool stop;

public:
    explicit FiberPool(size_t thread_count);
    ~FiberPool();

    template<class F>
    void enqueue(F task);
};

#include "fiber.tpp" // テンプレートメソッドの実装を含む

#endif
