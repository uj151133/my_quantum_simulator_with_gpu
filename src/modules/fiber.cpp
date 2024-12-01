#include "fiber.hpp"

FiberPool::FiberPool(size_t thread_count) : stop(false) {
    for (size_t i = 0; i < thread_count; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<boost::fibers::mutex> lock(queue_mutex);
                    condition.wait(lock, [this] {
                        return stop || !tasks.empty();
                    });

                    if (stop && tasks.empty()) {
                        return;
                    }

                    task = std::move(tasks.front());
                    tasks.pop();
                }

                task();
            }
        });
    }
}

FiberPool::~FiberPool() {
    {
        std::unique_lock<boost::fibers::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();

    for (auto& worker : workers) {
        worker.join();
    }
}
