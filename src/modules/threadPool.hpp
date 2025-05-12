#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP
#include <tbb/task_group.h>
#include <tbb/task_arena.h>
#include <functional>
#include <future>
#include <thread>

class ThreadPool {
    public:
        explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
        ~ThreadPool();
    
        // タスクを追加して結果を非同期で取得するための std::future を返す
        template <typename Func, typename... Args>
        auto enqueue(Func&& func, Args&&... args) -> std::future<decltype(func(args...))>;
    
        // タスクをすべて終了するまで待機（必要な場合のみ明示的に呼び出す）
        void wait();
    
    private:
        tbb::task_arena arena;  // TBBのスレッドプール管理用
        tbb::task_group group;  // タスク管理用
    };
    
    // グローバルなスレッドプールインスタンスの宣言
    extern ThreadPool threadPool;
    
    // テンプレート関数の定義（ヘッダ内に記述する必要があるため）
    template <typename Func, typename... Args>
    auto ThreadPool::enqueue(Func&& func, Args&&... args) -> std::future<decltype(func(args...))> {
        using ReturnType = decltype(func(args...));
    
        // タスクを std::packaged_task にラップ
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<Func>(func), std::forward<Args>(args)...)
        );
    
        std::future<ReturnType> result = task->get_future();
    
        // arena.execute を使用してタスクをスケジュール
        arena.execute([task, this]() {
            group.run([task]() { (*task)(); });
        });
    
        return result;
    }
    
    #endif