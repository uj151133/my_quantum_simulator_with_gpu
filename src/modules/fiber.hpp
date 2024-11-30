#pragma once

#include <boost/fiber/all.hpp>
#include <boost/asio.hpp>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

namespace parallel {

class FiberPool {
public:
    // シングルトンインスタンスを取得
    static FiberPool& getInstance(size_t threadCount = std::thread::hardware_concurrency());

    // ファイバータスクを追加
    template<typename Func>
    void enqueue(Func&& task);

    // すべてのファイバーの完了を待つ
    void wait();

    // デストラクタ
    ~FiberPool();

private:
    // コンストラクタは privateで、シングルトンパターンを実現
    explicit FiberPool(size_t threadCount);

    // ワークステアリングを実装するワーカースレッド関数
    void workerThread();

    // スレッドプール
    std::vector<std::thread> threads;

    // タスクキュー
    std::queue<std::function<void()>> tasks;

    // 同期プリミティブ
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;

    // 実行中のファイバー数
    std::atomic<size_t> activeFibers;

    // コンテキストスイッチングのためのI/Oサービス
    boost::asio::io_context ioContext;
    std::unique_ptr<boost::asio::io_context::work> work;
};

template<typename Func>
void FiberPool::enqueue(Func&& task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.emplace(std::forward<Func>(task));
    }
    condition.notify_one();
}

} 