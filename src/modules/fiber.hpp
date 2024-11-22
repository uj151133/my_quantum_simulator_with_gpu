// FiberPool.hpp
#ifndef FIBER_HPP
#define FIBER_HPP

#include <boost/fiber/all.hpp>
#include <vector>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

class FiberPool {
public:
    explicit FiberPool(size_t threadCount = std::thread::hardware_concurrency());
    ~FiberPool();

    // Delete copy constructor and assignment operator
    FiberPool(const FiberPool&) = delete;
    FiberPool& operator=(const FiberPool&) = delete;

    // タスクをキューに追加
    template <typename F>
    void enqueue(F&& task);

    // プールを停止してリソースを解放
    void stop();

private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> taskQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondition;
    std::atomic<bool> stopFlag;

    void workerThread();
};

#include "fiber.tpp"

#endif