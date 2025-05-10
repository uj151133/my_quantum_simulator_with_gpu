#include "threadPool.hpp"
ThreadPool threadPool;

// コンストラクタ：指定されたスレッド数でタスクアリーナを初期化
ThreadPool::ThreadPool(size_t numThreads) : arena(numThreads) {}

// デストラクタ：タスクの完了を待機
ThreadPool::~ThreadPool() {
    wait(); // すべてのタスクの終了を待機
}

// タスクの完了を待機
void ThreadPool::wait() {
    arena.execute([this]() { group.wait(); });
}