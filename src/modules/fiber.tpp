#ifndef FIBER_TPP
#define FIBER_TPP

template <typename F>
void FiberPool::enqueue(F&& task) {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        taskQueue.emplace(std::forward<F>(task));
    }
    queueCondition.notify_one();
}

#endif