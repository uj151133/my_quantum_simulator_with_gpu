#include "importer.hpp"



namespace aiinfer {

// 簡易大文字化
static inline string upper_copy(string s){
    for (auto& c: s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

// 特徴次元（既定 128, 環境変数 QMDD_SCHED_FEAT_DIM で上書き可）
static inline int feature_dim_from_env(){
    if (const char* v = getenv("QMDD_SCHED_FEAT_DIM")){
        try { int d = stoi(v); return max(16, d); } catch(...) {}
    }
    return 128;
}

// 対角タグ判定（ローカル）
static inline bool is_diag_tag(const std::string& tagU){
    return tagU=="RZ"||tagU=="U1"||tagU=="P"||tagU=="S"||tagU=="T"||tagU=="Z"||
           tagU=="CZ"||tagU=="CP"||tagU=="CRZ"||tagU=="RZZ";
}

// 1ゲート特徴（D次元）
static void make_gate_feature(const Core& op, int idx, int D, float* out){
    fill(out, out + D, 0.0f);
    const string tagU = upper_copy(op.tag);
    // tag ハッシュ
    size_t h = hash<string>{}(tagU);
    out[static_cast<int>(h % static_cast<size_t>(D))] += 0.5f;
    // 最小 qubit
    if (!op.qubits.empty()){
        int mq = *std::min_element(op.qubits.begin(), op.qubits.end());
        out[(mq % D + D) % D] += 0.5f;
    }
    // 対角フラグ
    out[D-1] = is_diag_tag(tagU) ? 1.0f : 0.0f;
    // 位置エンコーディング
    out[(idx * 7) % D] += 0.25f;
}

// 実装本体
struct SchedulerONNX::Impl {
    Ort::Env env;
    Ort::SessionOptions opts;
    Ort::Session session;

    static constexpr const char* kInputName  = "input";
    static constexpr const char* kOutputName = "perm";

    explicit Impl(const string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "qmdd_sched"),
      opts(),
      session(nullptr)
    {
        if (!fs::exists(model_path))
            throw runtime_error("ONNX model not found: " + model_path);
        session = Ort::Session(env, model_path.c_str(), opts);
    }

    vector<int> run(const vector<Core>& ops){
        const int64_t N = static_cast<int64_t>(ops.size());
        if (N <= 0) return {};

        const int64_t D = feature_dim_from_env();
        vector<float> feats(static_cast<size_t>(N * D), 0.0f);
        for (int64_t i=0;i<N;++i){
            make_gate_feature(ops[static_cast<size_t>(i)], static_cast<int>(i), static_cast<int>(D),
                              feats.data() + static_cast<size_t>(i * D));
        }

        // 入力 [1, N, D]
        array<int64_t,3> dims = {1, N, D};
        Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mi, feats.data(), feats.size(), dims.data(), dims.size()
        );

        const char* in_names[]  = { kInputName };
        const char* out_names[] = { kOutputName };

        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                   in_names,  &in_tensor, 1,
                                   out_names, 1);
        if (outputs.size() != 1 || !outputs[0].IsTensor())
            throw std::runtime_error("Invalid ONNX output: expect 1 tensor");

        Ort::Value& out = outputs[0];
        Ort::TensorTypeAndShapeInfo ti = out.GetTensorTypeAndShapeInfo();
        if (ti.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            throw std::runtime_error("Invalid ONNX output dtype: expect int64");

        vector<int64_t> shape = ti.GetShape();
        if (!((shape.size()==1 && shape[0]==N) || (shape.size()==2 && shape[0]==1 && shape[1]==N)))
            throw std::runtime_error("Invalid ONNX output shape: expect [N] or [1,N]");

        const int64_t* pv = out.GetTensorData<int64_t>();
        vector<int> perm(static_cast<size_t>(N));
        for (int64_t i=0;i<N;++i) perm[static_cast<size_t>(i)] = static_cast<int>(pv[i]);

        // 順列検証
        vector<bool> used(static_cast<size_t>(N), false);
        for (int v : perm) {
            if (v < 0 || v >= N || used[static_cast<size_t>(v)])
                throw std::runtime_error("ONNX output is not a valid permutation");
            used[static_cast<size_t>(v)] = true;
        }
        return perm;
    }
};

SchedulerONNX::SchedulerONNX(const std::string& modelPath)
: impl_(std::make_unique<Impl>(modelPath)) {}

SchedulerONNX::~SchedulerONNX() = default;

vector<int> SchedulerONNX::propose_order(const vector<Core>& ops){
    std::vector<Core> nops = ops;
    for (auto& c : nops) { Core cc = c; cc.normalize(); c = std::move(cc); }
    return this->impl_->run(nops);
}

}