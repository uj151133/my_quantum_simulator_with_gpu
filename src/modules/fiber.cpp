// FiberPool.cpp
#include "fiber.hpp"

FiberPool::FiberPool(size_t threadCount)
    : stopFlag(false) {
    for (size_t i = 0; i < threadCount; ++i) {
        threads.emplace_back([this]() { this->workerThread(); });
    }
}

FiberPool::~FiberPool() {
    stop();
}

void FiberPool::stop() {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        stopFlag = true;
    }
    queueCondition.notify_all();
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void FiberPool::workerThread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCondition.wait(lock, [this]() {
                return stopFlag || !taskQueue.empty();
            });
            if (stopFlag && taskQueue.empty()) {
                return;
            }
            task = std::move(taskQueue.front());
            taskQueue.pop();
        }
        // タスクを実行
        boost::fibers::fiber(std::move(task)).detach();
    }
}