#include "threadPool.hpp"

// boost::asio::thread_pool threadPool(std::thread::hardware_concurrency());


tbb::global_control* global_tbb_control = nullptr;

void initialize_tbb_thread_pool(int num_threads) {
    if (!global_tbb_control) {
        global_tbb_control = new tbb::global_control(
            tbb::global_control::max_allowed_parallelism, num_threads
        );
    }
}

void finalize_tbb_thread_pool() {
    delete global_tbb_control;
    global_tbb_control = nullptr;
}