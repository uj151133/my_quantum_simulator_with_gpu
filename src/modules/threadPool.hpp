#pragma once
#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

// #include <thread>
// #include <boost/asio/thread_pool.hpp>
// #include <boost/asio.hpp>

// extern boost::asio::thread_pool threadPool;

#include <tbb/global_control.h>

extern tbb::global_control* global_tbb_control;

void initialize_tbb_thread_pool(int num_threads);

void finalize_tbb_thread_pool();



#endif
