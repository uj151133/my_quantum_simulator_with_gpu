#include "threadPool.hpp"

ThreadPool threadPool;

namespace {
thread_local bool g_isFiberWorkerThread = false;
}

bool ThreadPool::isFiberWorkerThread_() {
    return g_isFiberWorkerThread;
}

ThreadPool::ThreadPool(size_t numThreads)
    : workerCount_(numThreads == 0 ? 1 : numThreads) {
    startWorkers_(workerCount_);
}

void ThreadPool::shutdown() {
    stopWorkers_();
    wait();
}

ThreadPool::~ThreadPool() {
    shutdown();
}

bool ThreadPool::postFiber(FiberJob job) {
    if (!running_.load(std::memory_order_acquire)) return false;
    return queue_.push(std::move(job)) == boost::fibers::channel_op_status::success;
}

void ThreadPool::wait() {
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

void ThreadPool::startWorkers_(size_t n) {
    running_.store(true, std::memory_order_release);

    {
        std::lock_guard<std::mutex> lk(startMx_);
        started_ = 0;
    }

    workers_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        workers_.emplace_back([this]() { workerLoop_(); });
    }

    // 全ワーカーがスケジューラ初期化を終えるまで待つ（work_stealing の前提を満たす）
    std::unique_lock<std::mutex> lk(startMx_);
    startCv_.wait(lk, [this]() { return started_ == workerCount_; });
}

void ThreadPool::stopWorkers_() {
    const bool wasRunning = running_.exchange(false, std::memory_order_acq_rel);
    if (!wasRunning) return;
    queue_.close();
}

void ThreadPool::markWorkerStarted_() {
    std::lock_guard<std::mutex> lk(startMx_);
    ++started_;
    if (started_ == workerCount_) {
        startCv_.notify_all();
    }
}

void ThreadPool::workerLoop_() {
    g_isFiberWorkerThread = true;

    // 各OSスレッドに一度だけ設定
    thread_local bool installed = false;
    if (!installed) {
        installed = true;

        const std::uint32_t participants =
            static_cast<std::uint32_t>(workerCount_ > 0 ? workerCount_ : 1);

        // work stealing を有効化（idle時はsuspend）
        boost::fibers::use_scheduling_algorithm<boost::fibers::algo::work_stealing>(participants, true);

        // 初期化完了を通知（全員揃うのを待つ）
        markWorkerStarted_();
    }

    while (running_.load(std::memory_order_acquire)) {
        FiberJob job;
        const auto st = queue_.pop(job);
        if (st != boost::fibers::channel_op_status::success) break;

        // job自体は “そのまま実行” でまず安定させる（内部でspawnしたfiberはsteal対象になり得る）
        job();
    }
}