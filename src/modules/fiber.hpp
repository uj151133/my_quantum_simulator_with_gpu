#ifndef FIBER_HPP
#define FIBER_HPP

#include <boost/fiber/all.hpp>
#include <deque>

class CustomScheduler : public boost::fibers::algo::algorithm {
public:
    void awakened(boost::fibers::context *ctx) noexcept override;
    boost::fibers::context* pick_next() noexcept override;
    bool has_ready_fibers() const noexcept override;
    void suspend_until(std::chrono::steady_clock::time_point const&) noexcept override;
    void notify() noexcept override;

private:
    std::deque<boost::fibers::context*> ready_queue;
};

#endif
