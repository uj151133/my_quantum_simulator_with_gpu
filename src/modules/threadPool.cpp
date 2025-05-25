#include "threadPool.hpp"

boost::asio::thread_pool threadPool(std::thread::hardware_concurrency());