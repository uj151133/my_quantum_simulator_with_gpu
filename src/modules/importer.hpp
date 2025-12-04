#ifndef IMPORTER_HPP
#define IMPORTER_HPP

static constexpr const char* SCHEDULER_MODEL_PATH = "AI/exports/אָדָם.onnx";
static constexpr const char* FUSER_MODEL_PATH     = "AI/exports/חַוָּה.onnx";

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <onnxruntime_cxx_api.h>
#include "../common/Core.hpp"

using namespace std;

namespace fs = filesystem;

namespace aiinfer {

// モデル1（順序提案）ONNX 推論ラッパ
// 入力: "input" [1,N,D] (float32), 出力: "perm" [N] (int64, 0..N-1 の順列)
class SchedulerONNX {
public:
    explicit SchedulerONNX(const string& modelPath = ::SCHEDULER_MODEL_PATH);
    ~SchedulerONNX();
    vector<int> predict(const vector<Core>& ops);

private:
    struct Impl;
    unique_ptr<Impl> impl_;
};

}

#endif