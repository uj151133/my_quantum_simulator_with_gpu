#ifndef FIBER_TPP
#define FIBER_TPP

#include "fiber.hpp"

template<class F>
void FiberPool::enqueue(F task) {
    {
        std::unique_lock<boost::fibers::mutex> lock(queue_mutex);
        tasks.emplace(task);
    }
    condition.notify_one();
}

#endif // FIBER_TPP
