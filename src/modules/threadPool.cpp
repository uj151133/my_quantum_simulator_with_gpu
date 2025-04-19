#include "threadPool.hpp"

boost::asio::thread_pool g_thread_pool(std::thread::hardware_concurrency());
