#pragma once
#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <thread>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio.hpp>

extern boost::asio::thread_pool g_thread_pool;

#endif
