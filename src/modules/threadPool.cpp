#include "threadPool.hpp"
#include "../common/parameter.hpp"
#include <algorithm>

ThreadPool threadPool;

namespace {
thread_local bool g_isFiberWorkerThread = false;
}

bool ThreadPool::isFiberWorkerThread_() {
    return g_isFiberWorkerThread;
}

ThreadPool::ThreadPool(size_t numThreads)
    : workerCount_(numThreads) {}   // 0 = 起動時に config / hardware_concurrency から決める

void ThreadPool::ensureStarted_() {
    std::call_once(this->startOnce_, [this]() {
        size_t n = this->workerCount_;
        if (n == 0) n = static_cast<size_t>(std::max(0, PARAMETER.parallelism.threads));
        if (n == 0) n = std::thread::hardware_concurrency();
        if (n == 0) n = 1;
        this->workerCount_ = n;
        this->startWorkers_(n);
    });
}

void ThreadPool::shutdown() {
    this->stopWorkers_();
    this->wait();
}

ThreadPool::~ThreadPool() {
    this->shutdown();
}

bool ThreadPool::postFiber(FiberJob job) {
    this->ensureStarted_();
    if (!this->running_.load(std::memory_order_acquire)) return false;
    return this->queue_.push(std::move(job)) == boost::fibers::channel_op_status::success;
}

void ThreadPool::wait() {
    for (auto& t : this->workers_) {
        if (t.joinable()) t.join();
    }
    this->workers_.clear();
}

void ThreadPool::startWorkers_(size_t n) {
    this->running_.store(true, std::memory_order_release);

    {
        std::lock_guard<std::mutex> lk(this->startMx_);
        this->started_ = 0;
    }

    this->workers_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        this->workers_.emplace_back([this]() { this->workerLoop_(); });
    }

    // 全ワーカーがスケジューラ初期化を終えるまで待つ（work_stealing の前提を満たす）
    std::unique_lock<std::mutex> lk(this->startMx_);
    this->startCv_.wait(lk, [this]() { return this->started_ == this->workerCount_; });
}

void ThreadPool::stopWorkers_() {
    const bool wasRunning = this->running_.exchange(false, std::memory_order_acq_rel);
    if (!wasRunning) return;
    this->queue_.close();
}

void ThreadPool::markWorkerStarted_() {
    std::lock_guard<std::mutex> lk(this->startMx_);
    ++this->started_;
    if (this->started_ == this->workerCount_) {
        this->startCv_.notify_all();
    }
}

void ThreadPool::workerLoop_() {
    g_isFiberWorkerThread = true;

    // 各OSスレッドに一度だけ設定
    thread_local bool installed = false;
    if (!installed) {
        installed = true;

        const std::uint32_t participants =
            static_cast<std::uint32_t>(this->workerCount_ > 0 ? this->workerCount_ : 1);

        boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(participants, false);

        // 初期化完了を通知（全員揃うのを待つ）
        this->markWorkerStarted_();
    }

    while (this->running_.load(std::memory_order_acquire)) {
        FiberJob job;
        const auto st = this->queue_.pop(job);
        if (st != boost::fibers::channel_op_status::success) break;

        // job自体は “そのまま実行” でまず安定させる（内部でspawnしたfiberはsteal対象になり得る）
        job();
    }
}