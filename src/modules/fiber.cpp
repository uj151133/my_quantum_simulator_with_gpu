#include "fiber.hpp"

void CustomScheduler::awakened(boost::fibers::context *ctx) noexcept {
    ready_queue.push_back(ctx);
}

boost::fibers::context* CustomScheduler::pick_next() noexcept {
    if (ready_queue.empty()) {
        return nullptr;
    }
    boost::fibers::context* ctx = ready_queue.front();
    ready_queue.pop_front();
    return ctx;
}

bool CustomScheduler::has_ready_fibers() const noexcept {
    return !ready_queue.empty();
}

void CustomScheduler::suspend_until(std::chrono::steady_clock::time_point const&) noexcept {}

void CustomScheduler::notify() noexcept {}
