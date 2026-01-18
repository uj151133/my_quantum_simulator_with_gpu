#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <boost/fiber/all.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    ~ThreadPool();
    void shutdown();

    using FiberJob = std::function<void()>;

    // Fiberジョブ投入（ジョブはワーカースレッド上で実行される）
    bool postFiber(FiberJob job);

    // ワーカー停止＆join（明示的に止めたい場合用。デストラクタでも呼ばれる）
    void wait();

    // 最上位実行用：job をワーカー上で開始して結果を返す
    template <class F>
    auto submitFiber(F&& f) -> std::future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;

        auto prom = std::make_shared<std::promise<R>>();
        auto fut  = prom->get_future();

        // worker自身から submitFiber().get() するとOSスレッドブロックになり得るので回避
        if (isFiberWorkerThread_()) {
            try {
                if constexpr (std::is_void_v<R>) {
                    std::forward<F>(f)();
                    prom->set_value();
                } else {
                    prom->set_value(std::forward<F>(f)());
                }
            } catch (...) {
                prom->set_exception(std::current_exception());
            }
            return fut;
        }

        const bool ok = postFiber([prom, fn = std::forward<F>(f)]() mutable {
            try {
                if constexpr (std::is_void_v<R>) {
                    fn();
                    prom->set_value();
                } else {
                    prom->set_value(fn());
                }
            } catch (...) {
                prom->set_exception(std::current_exception());
            }
        });

        if (!ok) {
            prom->set_exception(std::make_exception_ptr(
                std::runtime_error("submitFiber: threadPool is not running")
            ));
        }

        return fut;
    }

private:
    std::atomic<bool> running_{false};

    boost::fibers::buffered_channel<FiberJob> queue_{1 << 16};

    size_t workerCount_{0};
    std::vector<std::thread> workers_;

    // 全ワーカーが work_stealing を初期化し終わるまで待つためのバリア
    std::mutex startMx_;
    std::condition_variable startCv_;
    size_t started_{0};

    void startWorkers_(size_t n);
    void stopWorkers_();
    void workerLoop_();

    void markWorkerStarted_();

    static bool isFiberWorkerThread_();
};

extern ThreadPool threadPool;

#endif