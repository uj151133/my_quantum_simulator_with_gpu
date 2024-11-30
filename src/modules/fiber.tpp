// このファイルは主にテンプレートのインライン実装のために使用
// 通常、.tppファイルは#includeされることを想定

#ifndef FIBER_TPP
#define FIBER_TPP

#include "fiber.hpp"

namespace parallel {

template<typename Func>
void FiberPool::enqueue(Func&& task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        
        // タスクキューに追加
        tasks.emplace([task = std::forward<Func>(task)]() {
            // ラムダ内でタスクを実行
            task();
        });
    }
    
    // ワーカースレッドに通知
    condition.notify_one();
}

} // namespace parallel

#endif