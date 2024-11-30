#include "fiber.hpp"

namespace parallel {

FiberPool& FiberPool::getInstance(size_t threadCount) {
    static FiberPool instance(threadCount);
    return instance;
}

FiberPool::FiberPool(size_t threadCount) : stop(false), activeFibers(0) {
    // I/Oコンテキストの作業を設定してスレッドがすぐに終了しないようにする
    work = std::make_unique<boost::asio::io_context::work>(ioContext);

    // ワーカースレッドを起動
    for (size_t i = 0; i < threadCount; ++i) {
        threads.emplace_back([this] { workerThread(); });
    }
}

void FiberPool::workerThread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            
            // タスクが来るまで待機
            condition.wait(lock, [this] { 
                return stop || !tasks.empty(); 
            });

            // 停止フラグが立っていて、タスクキューが空の場合は終了
            if (stop && tasks.empty()) {
                return;
            }

            // タスクを取得
            task = std::move(tasks.front());
            tasks.pop();
        }

        // ワークステアリング：現在のスレッドでファイバーを実行
        boost::fibers::fiber([&task, this]() {
            ++activeFibers;
            try {
                task();
            } catch (const std::exception& e) {
                // エラーハンドリング
                std::cerr << "Fiber task failed: " << e.what() << std::endl;
            }
            --activeFibers;
        }).detach();
    }
}

void FiberPool::wait() {
    // すべてのファイバーの完了を待つ
    while (activeFibers > 0) {
        std::this_thread::yield();
    }
}

FiberPool::~FiberPool() {
    // すべてのスレッドを停止
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();

    // すべてのワーカースレッドを結合
    for (auto& thread : threads) {
        thread.join();
    }
}

} // namespace parallel