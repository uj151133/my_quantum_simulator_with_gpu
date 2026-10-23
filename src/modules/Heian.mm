#import <Metal/Metal.h>
#include <cstring>
#include <cstdio>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <boost/fiber/operations.hpp>   // GPU 待ちで他の fiber に譲る
#include "../models/arrayLeaf.hpp"
#include "../models/qmdd.hpp"
#include "backend.hpp"
#include "Heian.h"      // 公開 API と Metal 固有の型（GPUInputHeader / NormValHost / makeHeader）

// このファイルは ARC 前提（CMakeLists.txt で -fobjc-arc を付けている）。
// MRC だと newBuffer の +1 とハンドル化の retain が二重になり、バッファが一切解放されない。
#if !__has_feature(objc_arc)
#error "Heian.mm は ARC（-fobjc-arc）でビルドすること"
#endif


struct MetalContext {
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> queue;
};

static MetalContext& getMetalContext() {
    static MetalContext ctx = [] {
        MetalContext c{};
        c.device = MTLCreateSystemDefaultDevice();
        NSURL* libURL = [NSURL fileURLWithPath:@"/Users/mitsuishikaito/my_quantum_simulator_with_gpu/build/Heian.metallib"];
        NSError* err = nil;
        c.library = [c.device newLibraryWithURL:libURL error:&err];
        c.queue = [c.device newCommandQueue];
        return c;
    }();
    return ctx;
}

// === 使い回しスクラッチ（作業スロットごと）=====================================
// Metal も per-op の newBuffer をやめ、固定サイズの一時バッファは使い回す。
// スロット（下の acquireSlot）として 1 演算が完了まで占有するので、使い回しは安全。
// ARC で消えないよう strong 参照（id）で保持する。
struct MtlScratch {
    id<MTLBuffer> outId   = nil;   // sizeof(uint64_t)
    id<MTLBuffer> outCoef = nil;   // sizeof(NormValHost)
    // runNormalizePass / runHash2Pass 用（毎op 生成していたものを使い回す）
    id<MTLBuffer> groupVal = nil;  // grow-only（L2 正規化の部分和）
    id<MTLBuffer> normTotal = nil; // 内容 total
    id<MTLBuffer> normGroup = nil; // 内容 groups
    id<MTLBuffer> partial  = nil;  // grow-only
    id<MTLBuffer> hashTotal = nil; // 内容 total
    id<MTLBuffer> hashGroup = nil; // 内容 groups
    // runMulAny2 用
    id<MTLBuffer> dimBuf = nil;     // 内容 dim
    id<MTLBuffer> hdrA   = nil;     // 内容 GPUInputHeader
    id<MTLBuffer> hdrB   = nil;
    id<MTLBuffer> svHdr  = nil;
    id<MTLBuffer> svHdrB = nil;
    id<MTLBuffer> edgesA = nil;     // grow-only
    id<MTLBuffer> edgesB = nil;     // grow-only
    id<MTLBuffer> denseARe = nil;   // grow-only
    id<MTLBuffer> denseAIm = nil;
    id<MTLBuffer> denseBRe = nil;
    id<MTLBuffer> denseBIm = nil;
};

// dense 化できる上限。シェーダは tid を uint で索引しているので 2^32 要素は扱えず、
// メモリにも載らない。uint32 が黙ってラップして不正なバッファを作るのを防ぐ。
static inline uint32_t checkedTotal(const char* op, uint64_t dim) {
    const uint64_t total = dim * dim;
    if (total > 0x7FFFFFFFull) {
        fprintf(stderr,
                "FATAL @%s: dense block too large: dim=%llu -> %llu elements. "
                "config.yaml の parallelism.GPU を上げて GPU 起動を浅くしてください。\n",
                op, (unsigned long long)dim, (unsigned long long)total);
        fflush(stderr);
        abort();
    }
    return (uint32_t)total;
}

// === 作業スロットのプール ===================================================
// GPU 完了待ちでファイバーを譲るようにすると、ファイバーは別の OS スレッドで
// 再開しうる。thread_local のままだと自分の仕事とは別のスクラッチを触るので、
// プールから貸し出す形にして同時実行数に上限を設ける（＝背圧）。
// CUDA 側（Heian.cu の GpuSlot）と同じ考え方。
static int gpuSlotCap() {
    static int cap = [] {
        const char* e = getenv("QMDD_GPU_SLOTS");
        int v = e ? atoi(e) : 64;
        return v > 0 ? v : 64;
    }();
    return cap;
}

static std::mutex               g_slotMx;
static std::vector<MtlScratch*> g_freeSlots;
static int                      g_slotsMade = 0;

static MtlScratch* acquireSlot() {
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(g_slotMx);
            if (!g_freeSlots.empty()) {
                MtlScratch* s = g_freeSlots.back();
                g_freeSlots.pop_back();
                return s;
            }
            if (g_slotsMade < gpuSlotCap()) { ++g_slotsMade; return new MtlScratch(); }
        }
        boost::this_fiber::yield();   // 上限まで飛んでいる。譲って待つ。
    }
}
static void releaseSlot(MtlScratch* s) {
    std::lock_guard<std::mutex> lk(g_slotMx);
    g_freeSlots.push_back(s);
}
struct SlotGuard {
    MtlScratch* s;
    SlotGuard() : s(acquireSlot()) {}
    ~SlotGuard() { releaseSlot(s); }
    SlotGuard(const SlotGuard&) = delete;
    SlotGuard& operator=(const SlotGuard&) = delete;
    MtlScratch& operator*()  const { return *s; }
    MtlScratch* operator->() const { return s; }
};

// コマンドバッファの完了を、OS スレッドを止めずに待つ。
// waitUntilCompleted はスレッドをブロックするので、そのスレッドに載っている
// 他のファイバーも道連れで止まってしまう。CUDA 側 waitStream と同じ役割。
static void waitCmd(id<MTLCommandBuffer> cmd) {
    [cmd commit];
    while (cmd.status != MTLCommandBufferStatusCompleted &&
           cmd.status != MTLCommandBufferStatusError) {
        boost::this_fiber::yield();
    }
}

// バッファの所有権を呼び出し側（ArrayLeaf）へ渡す。releaseGpuBuffer と対になる。
static inline void* handOverBuffer(id<MTLBuffer> buf) {
    return (__bridge_retained void*)buf;
}

// === GPU 出力バッファのプール（サイズ別フリーリスト）===========================
// 毎op の newBufferWithLength をやめ、解放されたバッファを使い回す。
// ArrayLeaf の releaseGpuBuffer はここに返す（解放しない）→ メモリ churn を止める。
// CUDA 側（Heian.cu の poolAlloc / poolFree / poolReclaimLocked）と同じ構成。
//
// プールが抱えるのは「retain 済みの生ハンドル」。ARC 下で id<MTLBuffer> を
// C++ コンテナに入れられないので、handOverBuffer と同じ流儀で void* にして持つ。
static std::mutex g_poolMx;
static std::unordered_map<size_t, std::vector<void*>> g_freePool;  // bytes → 空きハンドル
static std::unordered_map<void*, size_t> g_ptrSize;                // ハンドル → bytes

// retain 済みハンドル → id<MTLBuffer>（所有権は移さず覗くだけ）
static inline id<MTLBuffer> peekBuffer(void* p) {
    return (__bridge id<MTLBuffer>)p;
}

// プールが抱えている空きバッファを全て手放す（g_poolMx 保持前提）。
// dim は深さごとに変わるためバケットはサイズ別に複数でき、あるサイズの空きは
// 別サイズの確保には使えない。空きがあるのに OOM するのを避けるための最後の手段。
static size_t poolReclaimLocked() {
    size_t freed = 0;
    for (auto& kv : g_freePool) {
        for (void* q : kv.second) {
            g_ptrSize.erase(q);
            id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)q;   // retain を手放す
            (void)b;
            freed += kv.first;
        }
        kv.second.clear();
    }
    return freed;
}

// 確保に失敗すると newBufferWithLength は nil を返す。そのまま渡すと
// カーネルが不正アクセスするので、ここで OOM と分かる形にして投げる。
// CUDA 側（Heian.cu poolAlloc）にも同じ機能がある。
static void* poolAlloc(id<MTLDevice> dev, NSUInteger len) {
    std::lock_guard<std::mutex> lk(g_poolMx);
    auto& bucket = g_freePool[(size_t)len];
    if (!bucket.empty()) {
        void* p = bucket.back();
        bucket.pop_back();
        return p;                       // retain 済みのまま返す
    }
    id<MTLBuffer> b = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
    if (b == nil) {
        // 別サイズのバケットに死蔵された分を返してから確保し直す。
        size_t freed = poolReclaimLocked();
        if (freed > 0) b = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
    }
    if (b == nil) {
        throw std::runtime_error(
            "GPU out of memory: " + std::to_string((size_t)len) + " bytes required"
            " (閾値 PARAMETER.parallelism.GPU を上げてください)");
    }
    void* p = handOverBuffer(b);        // retain してハンドル化
    g_ptrSize[p] = (size_t)len;
    return p;
}

static void poolFree(void* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_poolMx);
    auto it = g_ptrSize.find(p);
    if (it == g_ptrSize.end()) {        // プール外（保険）。retain を手放す
        id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)p;
        (void)b;
        return;
    }
    g_freePool[it->second].push_back(p);   // バケットに返す（解放しない）
}

// 確保のみ（従来の allocBufOrThrow 互換）。プールを通さない固定スクラッチ用。
static id<MTLBuffer> allocBufOrThrow(id<MTLDevice> dev, NSUInteger len) {
    id<MTLBuffer> b = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
    if (b == nil) {
        {
            std::lock_guard<std::mutex> lk(g_poolMx);
            if (poolReclaimLocked() > 0) {
                b = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
            }
        }
    }
    if (b == nil) {
        throw std::runtime_error(
            "GPU out of memory: " + std::to_string((size_t)len) + " bytes required"
            " (閾値 PARAMETER.parallelism.GPU を上げてください)");
    }
    return b;
}

// 固定サイズの共有バッファを「無ければ作る／足りなければ作り直す」で使い回す（grow-only）
static id<MTLBuffer> ensureSharedBuf(id<MTLDevice> dev, id<MTLBuffer>& slot, NSUInteger len) {
    if (slot == nil || slot.length < len) {
        slot = allocBufOrThrow(dev, len);
    }
    return slot;
}

// 上に加えてホストの内容を毎op 書き込む版（newBufferWithBytes の使い回し版）
static id<MTLBuffer> ensureSharedBytes(id<MTLDevice> dev, id<MTLBuffer>& slot, const void* src, NSUInteger len) {
    id<MTLBuffer> b = ensureSharedBuf(dev, slot, len);
    memcpy(b.contents, src, len);
    return b;
}

static id<MTLComputePipelineState> getPSO(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* fnName
) {
    static std::mutex mtx;
    static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* cache = nil;

    std::lock_guard<std::mutex> lock(mtx);

    if (!cache) {
        cache = [NSMutableDictionary new];
    }

    id<MTLComputePipelineState> cached = cache[fnName];
    if (cached) return cached;

    id<MTLFunction> fn = [library newFunctionWithName:fnName];
    if (!fn) {
        NSLog(@"getPSO: function not found: %@", fnName);
        return nil;
    }

    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        NSLog(@"getPSO: PSO creation failed for %@: %@", fnName, err);
        return nil;
    }

    cache[fnName] = pso;
    return pso;
}

static void runNormalizePass(
    MtlScratch& sc,
    id<MTLDevice> device,
    id<MTLLibrary> library,
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> outReBuf,
    id<MTLBuffer> outImBuf,
    id<MTLBuffer> outCoefBuf,
    uint32_t total
) {
    id<MTLComputePipelineState> p1PSO = getPSO(device, library, @"norm_pass1");
    id<MTLComputePipelineState> p2PSO = getPSO(device, library, @"norm_pass2");
    id<MTLComputePipelineState> p3PSO = getPSO(device, library, @"norm_apply");

    NSUInteger tg = p1PSO.maxTotalThreadsPerThreadgroup;
    NSUInteger groups = (total + tg - 1) / tg;
    uint32_t groups32 = (uint32_t)groups;

    // 毎op 生成をやめ、スロットのスクラッチを使い回す（grow-only / 内容書き込み）
    // L2 正規化では部分和（float 1本/グループ）だけあればよい。
    id<MTLBuffer> groupSumBuf = ensureSharedBuf(device, sc.groupVal, sizeof(float) * groups);
    id<MTLBuffer> totalBuf    = ensureSharedBytes(device, sc.normTotal, &total, sizeof(uint32_t));
    id<MTLBuffer> groupBuf    = ensureSharedBytes(device, sc.normGroup, &groups32, sizeof(uint32_t));

    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p1PSO];
        [enc setBuffer:outReBuf offset:0 atIndex:0];
        [enc setBuffer:outImBuf offset:0 atIndex:1];
        [enc setBuffer:groupSumBuf offset:0 atIndex:2];
        [enc setBuffer:totalBuf offset:0 atIndex:3];

        MTLSize grid = MTLSizeMake(groups * tg, 1, 1);
        MTLSize tgs = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
    }

    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p2PSO];
        [enc setBuffer:groupSumBuf offset:0 atIndex:0];
        [enc setBuffer:outCoefBuf offset:0 atIndex:1];
        [enc setBuffer:groupBuf offset:0 atIndex:2];

        MTLSize grid = MTLSizeMake(1, 1, 1);
        MTLSize tgs = MTLSizeMake(1, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
    }

    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p3PSO];
        [enc setBuffer:outReBuf offset:0 atIndex:0];
        [enc setBuffer:outImBuf offset:0 atIndex:1];
        [enc setBuffer:outCoefBuf offset:0 atIndex:2];
        [enc setBuffer:totalBuf offset:0 atIndex:3];

        MTLSize grid = MTLSizeMake(total, 1, 1);
        MTLSize tgs = MTLSizeMake(p3PSO.maxTotalThreadsPerThreadgroup, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
    }
}

void runHash2Pass(
    MtlScratch& sc,
    id<MTLDevice> device,
    id<MTLLibrary> library,
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> outReBuf,
    id<MTLBuffer> outImBuf,
    id<MTLBuffer> outIdBuf,
    uint32_t total,
    id<MTLBuffer> outCoefBuf     // 成分ごとのゼロ判定を下ろす先（正規化後に見る）
) {
    id<MTLComputePipelineState> p1PSO = getPSO(device, library, @"hash_pass1");
    id<MTLComputePipelineState> p2PSO = getPSO(device, library, @"hash_pass2");

    NSUInteger tg = p1PSO.maxTotalThreadsPerThreadgroup;
    NSUInteger groups = (total + tg - 1) / tg;
    uint32_t groups32 = (uint32_t)groups;

    // 毎op 生成をやめ、スロットのスクラッチを使い回す
    id<MTLBuffer> partialBuf = ensureSharedBuf(device, sc.partial, sizeof(uint64_t) * groups);
    id<MTLBuffer> totalBuf   = ensureSharedBytes(device, sc.hashTotal, &total, sizeof(uint32_t));
    id<MTLBuffer> groupBuf   = ensureSharedBytes(device, sc.hashGroup, &groups32, sizeof(uint32_t));

    // pass1
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p1PSO];
        [enc setBuffer:outReBuf   offset:0 atIndex:0];
        [enc setBuffer:outImBuf   offset:0 atIndex:1];
        [enc setBuffer:partialBuf offset:0 atIndex:2];
        [enc setBuffer:totalBuf   offset:0 atIndex:3];
        [enc setBuffer:outCoefBuf offset:0 atIndex:4];

        MTLSize grid = MTLSizeMake(groups * tg, 1, 1);
        MTLSize tgs  = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
    }

    // pass2
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:p2PSO];
        [enc setBuffer:partialBuf offset:0 atIndex:0];
        [enc setBuffer:outIdBuf   offset:0 atIndex:1];
        [enc setBuffer:groupBuf   offset:0 atIndex:2];

        MTLSize grid = MTLSizeMake(1, 1, 1);
        MTLSize tgs  = MTLSizeMake(1, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
    }
}

// === 後処理（ノルム → 正規化 → ハッシュ）の切り替え ===========================
// total が小さいときは融合版 1 本（起動 1 回）、大きいときは多カーネル版で SM を使い切る。
// Heian.cu の run_postprocess と同じ切り替え。
#ifndef QMDD_FUSE_MAX
#define QMDD_FUSE_MAX 256u
#endif
static constexpr NSUInteger POST_TG = 256;   // Heian.metal の POST_TG と一致させること

static void runPostprocess(
    MtlScratch& sc,
    id<MTLDevice> device,
    id<MTLLibrary> library,
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> outReBuf,
    id<MTLBuffer> outImBuf,
    id<MTLBuffer> outIdBuf,
    id<MTLBuffer> outCoefBuf,
    uint32_t total
) {
    if (total <= QMDD_FUSE_MAX) {
        // --- 融合版：起動 1 回 ------------------------------------------
        id<MTLComputePipelineState> pso = getPSO(device, library, @"postprocess_fused");
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:outReBuf   offset:0 atIndex:0];
        [enc setBuffer:outImBuf   offset:0 atIndex:1];
        [enc setBuffer:outCoefBuf offset:0 atIndex:2];
        [enc setBuffer:outIdBuf   offset:0 atIndex:3];
        [enc setBytes:&total length:sizeof(uint32_t) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(POST_TG, 1, 1)];
        [enc endEncoding];
        return;
    }
    // --- 多カーネル版 ------------------------------------------------------
    // ハッシュは必ず正規化「後」に取る（定数倍違いを同一キーにするため）
    runNormalizePass(sc, device, library, cmd, outReBuf, outImBuf, outCoefBuf, total);
    runHash2Pass(sc, device, library, cmd, outReBuf, outImBuf, outIdBuf, total, outCoefBuf);
}

// === 単発演算 ================================================================
// encode*Any2 はコマンドバッファに積むだけで、commit も待ちもしない。
// 単発（runSingleMetal）と従来バッチ（runBatchCommonMetal）の両方がこれを使う。
//
// ★ autoreleasepool と yield
//   autorelease pool はスレッドごとの LIFO なので、pool の途中で fiber を譲ると
//   同じスレッドの別 fiber と push/pop が入れ違い、他人のオブジェクトまで解放してしまう。
//   fiber が別スレッドで再開すれば pop が別スレッドで起きて壊れる。
//   そのため pool はエンコード（yield しない）だけを囲み、待ち（waitCmd）とスロットの
//   借用（acquireSlot）は pool の外で行う。コマンドバッファは pool の外の強参照で持つ。

enum BatchOpKind { BATCH_OP_MUL = 0, BATCH_OP_ADD = 1, BATCH_OP_KRON = 2 };

static const char* opName(BatchOpKind op) {
    return op == BATCH_OP_MUL ? "mul" : op == BATCH_OP_ADD ? "add" : "kron";
}

// 演算 1 件ぶんのオペランド（ホスト側の見え方）
struct MtlOperand {
    GPUInputHeader hdr{};
    const void*    edges     = nullptr;   // QMDD のときの GPUEdge 列
    size_t         edgeBytes = 0;
    id<MTLBuffer>  re = nil;              // ArrayLeaf の密バッファ（成分が全ゼロなら nil）
    id<MTLBuffer>  im = nil;
};

static MtlOperand toOperand(const GPUInput& in) {
    MtlOperand o;
    o.hdr = makeHeader(in);
    if (!in.qmdd.edges.empty()) {
        o.edges     = in.qmdd.edges.data();
        o.edgeBytes = in.qmdd.edges.size() * sizeof(GPUEdge);
    }
    if (in.kind == SonKind::ArrayLeaf) {
        o.re = (__bridge id<MTLBuffer>)in.sv.reHandle;
        o.im = (__bridge id<MTLBuffer>)in.sv.imHandle;
    }
    return o;
}

// 演算 1 件ぶんの出力先。re/im はプール、hash/coef は後処理の結果。
struct MtlOutputs {
    id<MTLBuffer> re   = nil;
    id<MTLBuffer> im   = nil;
    id<MTLBuffer> hash = nil;   // uint64_t
    id<MTLBuffer> coef = nil;   // NormValHost
};

// 結果の一辺（pad 込み）。kron は掛け算になるので 64 ビットで返し、checkedTotal で検査する。
static uint64_t resultDim(BatchOpKind op, const GPUInputHeader& a, const GPUInputHeader& b) {
    return op == BATCH_OP_KRON ? (uint64_t)effDimH(a) * effDimH(b) : (uint64_t)effDimH(a);
}

// mul / kron は root 重みをホストで掛ける（add はカーネル内で掛け済み）。CUDA 側と同じ。
static void applyRoot(BatchOpKind op, const GPUInput& A, const GPUInput& B, float& re, float& im) {
    if (op == BATCH_OP_ADD) return;
    const float rA = (float)A.root_re, iA = (float)A.root_im;
    const float rB = (float)B.root_re, iB = (float)B.root_im;
    const float r = rA * rB - iA * iB;
    const float m = rA * iB + iA * rB;
    const float nr = re * r - im * m;
    const float ni = re * m + im * r;
    re = nr; im = ni;
}

// 後処理の結果から出力を確定する。全ゼロの成分はプールに返す（CUDA 側と同じ扱い）。
// 両方ゼロ = ゼロ行列。係数も 0 にして、呼び出し側に edgeZero を返させる。
static void settleOutput(BatchOpKind op, const GPUInput& A, const GPUInput& B,
                         float re, float im, bool zeroRe, bool zeroIm,
                         void*& reH, void*& imH, float coef[2]) {
    if (zeroRe) { poolFree(reH); reH = nullptr; }
    if (zeroIm) { poolFree(imH); imH = nullptr; }
    if (zeroRe && zeroIm) { coef[0] = 0.0f; coef[1] = 0.0f; return; }
    applyRoot(op, A, B, re, im);
    coef[0] = re;
    coef[1] = im;
}

// 完了まで譲りながら待ち、失敗していたら例外にする。
static void waitCmdOrThrow(id<MTLCommandBuffer> cmd, const char* what) {
    waitCmd(cmd);
    if (cmd.status == MTLCommandBufferStatusError) {
        throw std::runtime_error(std::string("Metal ") + what + " failed: "
                                 + (cmd.error ? cmd.error.localizedDescription.UTF8String : "unknown"));
    }
}

static void encodeMulAny2(MtlScratch& sc, MetalContext& ctx, id<MTLCommandBuffer> cmd,
                          const MtlOperand& A, const MtlOperand& B, const MtlOutputs& out) {
    id<MTLDevice> device = ctx.device;
    const uint32_t dim   = effDimH(A.hdr);     // A,B 同 dim 前提（パディング後の実効サイズ）
    const uint32_t total = checkedTotal("mul", (uint64_t)dim);
    const size_t denseBytes = (size_t)total * sizeof(float);

    id<MTLBuffer> dimBuf    = ensureSharedBytes(device, sc.dimBuf, &dim, sizeof(uint32_t));
    id<MTLBuffer> hdrBufA   = ensureSharedBytes(device, sc.hdrA, &A.hdr, sizeof(GPUInputHeader));
    id<MTLBuffer> hdrBufB   = ensureSharedBytes(device, sc.hdrB, &B.hdr, sizeof(GPUInputHeader));
    id<MTLBuffer> edgesBufA = A.edgeBytes ? ensureSharedBytes(device, sc.edgesA, A.edges, A.edgeBytes) : nil;
    id<MTLBuffer> edgesBufB = B.edgeBytes ? ensureSharedBytes(device, sc.edgesB, B.edges, B.edgeBytes) : nil;

    // --- Stage 1: QMDD オペランドを dense 化（ArrayLeaf は既に dense なのでそのまま使う）---
    id<MTLComputePipelineState> matPSO = getPSO(device, ctx.library, @"materialize_any");
    const MTLSize matGrid = MTLSizeMake(total, 1, 1);
    const MTLSize matTgs  = MTLSizeMake(matPSO.maxTotalThreadsPerThreadgroup, 1, 1);

    id<MTLBuffer> aRe = A.re, aIm = A.im;
    id<MTLBuffer> bRe = B.re, bIm = B.im;
    auto materialize = [&](id<MTLBuffer> hdrBuf, id<MTLBuffer> edgesBuf, const MtlOperand& X,
                           id<MTLBuffer> dRe, id<MTLBuffer> dIm) {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:matPSO];
        [enc setBuffer:hdrBuf   offset:0 atIndex:0];
        [enc setBuffer:edgesBuf offset:0 atIndex:1];
        [enc setBuffer:X.re     offset:0 atIndex:2];
        [enc setBuffer:X.im     offset:0 atIndex:3];
        [enc setBuffer:dRe      offset:0 atIndex:4];
        [enc setBuffer:dIm      offset:0 atIndex:5];
        [enc setBuffer:dimBuf   offset:0 atIndex:6];
        [enc dispatchThreads:matGrid threadsPerThreadgroup:matTgs];
        [enc endEncoding];
    };
    if (A.hdr.kind == KIND_QMDD) {  // QMDDNode → dense 化
        aRe = ensureSharedBuf(device, sc.denseARe, denseBytes);
        aIm = ensureSharedBuf(device, sc.denseAIm, denseBytes);
        materialize(hdrBufA, edgesBufA, A, aRe, aIm);
    }
    if (B.hdr.kind == KIND_QMDD) {
        bRe = ensureSharedBuf(device, sc.denseBRe, denseBytes);
        bIm = ensureSharedBuf(device, sc.denseBIm, denseBytes);
        materialize(hdrBufB, edgesBufB, B, bRe, bIm);
    }

    // --- mul_any2 を「両方 SV 扱い（dense）」で実行：evalDD 無しの素直な dense matmul ---
    // materialize 済みの dense は既に実効サイズ（パディング適用後）なので、
    // pad を 1 に戻さないと evalInput が二重にパディングを掛けてしまう。
    // ArrayLeaf のまま渡すオペランドは展開していないので dim/pad を元のまま残す。
    // Heian.cu の svA/svB と同じ扱いにすること。
    GPUInputHeader svA{};
    svA.kind = KIND_SV; svA.root_re = 1.0f; svA.root_im = 0.0f;
    svA.dim = dim; svA.pad = 1u; svA.hasRe = 1u; svA.hasIm = 1u;
    GPUInputHeader svB = svA;
    if (A.hdr.kind == KIND_SV) { svA.dim = A.hdr.dim; svA.pad = A.hdr.pad; svA.hasRe = A.hdr.hasRe; svA.hasIm = A.hdr.hasIm; }
    if (B.hdr.kind == KIND_SV) { svB.dim = B.hdr.dim; svB.pad = B.hdr.pad; svB.hasRe = B.hdr.hasRe; svB.hasIm = B.hdr.hasIm; }
    id<MTLBuffer> svHdrBufA = ensureSharedBytes(device, sc.svHdr,  &svA, sizeof(GPUInputHeader));
    id<MTLBuffer> svHdrBufB = ensureSharedBytes(device, sc.svHdrB, &svB, sizeof(GPUInputHeader));

    id<MTLComputePipelineState> pso = getPSO(device, ctx.library, @"mul_any2");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:svHdrBufA offset:0 atIndex:0];
    [enc setBuffer:svHdrBufB offset:0 atIndex:1];
    [enc setBuffer:edgesBufA offset:0 atIndex:2];   // kind=SV なので未使用
    [enc setBuffer:edgesBufB offset:0 atIndex:3];   // 同上
    [enc setBuffer:aRe       offset:0 atIndex:4];
    [enc setBuffer:aIm       offset:0 atIndex:5];
    [enc setBuffer:bRe       offset:0 atIndex:6];
    [enc setBuffer:bIm       offset:0 atIndex:7];
    [enc setBuffer:out.re    offset:0 atIndex:8];
    [enc setBuffer:out.im    offset:0 atIndex:9];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1)];
    [enc endEncoding];

    runPostprocess(sc, device, ctx.library, cmd, out.re, out.im, out.hash, out.coef, total);
}

// add / kron は QMDD を dense 化せず、カーネルが evalInput で直接読む。違いはカーネル名と結果の一辺だけ。
static void encodeElementwiseAny2(BatchOpKind op, MtlScratch& sc, MetalContext& ctx, id<MTLCommandBuffer> cmd,
                                  const MtlOperand& A, const MtlOperand& B, const MtlOutputs& out) {
    id<MTLDevice> device = ctx.device;
    // pad 込みの実効サイズで数える（カーネル側も effDim で一辺を出している）
    const uint32_t total = checkedTotal(opName(op), resultDim(op, A.hdr, B.hdr));

    id<MTLBuffer> hdrBufA   = ensureSharedBytes(device, sc.hdrA, &A.hdr, sizeof(GPUInputHeader));
    id<MTLBuffer> hdrBufB   = ensureSharedBytes(device, sc.hdrB, &B.hdr, sizeof(GPUInputHeader));
    id<MTLBuffer> edgesBufA = A.edgeBytes ? ensureSharedBytes(device, sc.edgesA, A.edges, A.edgeBytes) : nil;
    id<MTLBuffer> edgesBufB = B.edgeBytes ? ensureSharedBytes(device, sc.edgesB, B.edges, B.edgeBytes) : nil;

    id<MTLComputePipelineState> pso =
        getPSO(device, ctx.library, op == BATCH_OP_ADD ? @"add_any2" : @"kron_any2");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:hdrBufA   offset:0 atIndex:0];
    [enc setBuffer:hdrBufB   offset:0 atIndex:1];
    [enc setBuffer:edgesBufA offset:0 atIndex:2];
    [enc setBuffer:edgesBufB offset:0 atIndex:3];
    [enc setBuffer:A.re      offset:0 atIndex:4];
    [enc setBuffer:A.im      offset:0 atIndex:5];
    [enc setBuffer:B.re      offset:0 atIndex:6];
    [enc setBuffer:B.im      offset:0 atIndex:7];
    [enc setBuffer:out.re    offset:0 atIndex:8];
    [enc setBuffer:out.im    offset:0 atIndex:9];
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1)];
    [enc endEncoding];

    runPostprocess(sc, device, ctx.library, cmd, out.re, out.im, out.hash, out.coef, total);
}

static void encodeOp(BatchOpKind op, MtlScratch& sc, MetalContext& ctx, id<MTLCommandBuffer> cmd,
                     const MtlOperand& A, const MtlOperand& B, const MtlOutputs& out) {
    if (op == BATCH_OP_MUL) encodeMulAny2(sc, ctx, cmd, A, B, out);
    else                    encodeElementwiseAny2(op, sc, ctx, cmd, A, B, out);
}

// 単発の 1 演算：スロットを借りて積み、譲りながら待って結果を読む。
static void runSingleMetal(BatchOpKind op, const GPUInput& A, const GPUInput& B,
                           void** outRe, void** outIm, int64_t* outId, float* outCoef) {
    auto& ctx = getMetalContext();
    SlotGuard slot;                     // 空きが無ければ yield するので pool の外で借りる

    const MtlOperand a = toOperand(A);
    const MtlOperand b = toOperand(B);
    const NSUInteger outBytes =
        (NSUInteger)checkedTotal(opName(op), resultDim(op, a.hdr, b.hdr)) * sizeof(float);

    void* reH = nullptr;
    void* imH = nullptr;
    try {
        reH = poolAlloc(ctx.device, outBytes);
        imH = poolAlloc(ctx.device, outBytes);
        MtlOutputs out;
        out.re   = peekBuffer(reH);
        out.im   = peekBuffer(imH);
        out.hash = ensureSharedBuf(ctx.device, slot->outId, sizeof(uint64_t));
        out.coef = ensureSharedBuf(ctx.device, slot->outCoef, sizeof(NormValHost));

        id<MTLCommandBuffer> cmd = nil;
        @autoreleasepool {
            cmd = [ctx.queue commandBuffer];
            encodeOp(op, *slot, ctx, cmd, a, b, out);
        }
        waitCmdOrThrow(cmd, opName(op));

        const NormValHost* nv = (const NormValHost*)out.coef.contents;
        float coef[2];
        settleOutput(op, A, B, nv->re, nv->im, nv->zeroRe != 0, nv->zeroIm != 0, reH, imH, coef);
        if (outRe)   *outRe = reH;      // プールから来た retain 済みハンドルをそのまま渡す
        if (outIm)   *outIm = imH;
        if (outId)   *outId = *(const int64_t*)out.hash.contents;
        if (outCoef) { outCoef[0] = coef[0]; outCoef[1] = coef[1]; }
    } catch (...) {
        poolFree(reH);
        poolFree(imH);
        throw;
    }
}

extern "C" void runMulAny2Wrapper(const GPUInput& A, const GPUInput& B,
                                  void** outRe, void** outIm, int64_t* outId, float* outCoef) {
    runSingleMetal(BATCH_OP_MUL, A, B, outRe, outIm, outId, outCoef);
}
extern "C" void runAddAny2Wrapper(const GPUInput& A, const GPUInput& B,
                                  void** outRe, void** outIm, int64_t* outId, float* outCoef) {
    runSingleMetal(BATCH_OP_ADD, A, B, outRe, outIm, outId, outCoef);
}
extern "C" void runKronAny2Wrapper(const GPUInput& A, const GPUInput& B,
                                   void** outRe, void** outIm, int64_t* outId, float* outCoef) {
    runSingleMetal(BATCH_OP_KRON, A, B, outRe, outIm, outId, outCoef);
}


// ===========================================================================
// === バッチ実行（従来経路）==================================================
// 融合版（runBatchFusedMetal）が使えない環境向け。全ジョブのエンコードを 1 つの
// コマンドバッファに積んで 1 回 commit する。往復が 1 回で済む点は融合版と同じ。
//
// 注意: パスは共有スクラッチを使い回す作りなので、同じコマンドバッファに
// 複数ジョブを積むとバッファを奪い合う。ジョブごとに MtlScratch を持たせる。
// ===========================================================================

static void runBatchCommonMetal(BatchOpKind op, BatchJob* jobs, int nJobs) {
    if (!jobs || nJobs <= 0) return;
    auto& ctx = getMetalContext();
    const size_t n = static_cast<size_t>(nJobs);

    // ジョブごとの専有スクラッチ。コマンドバッファの完了まで生かしておく。
    std::vector<MtlScratch> scr(n);
    std::vector<MtlOutputs> outs(n);
    std::vector<void*>      reH(n, nullptr);    // プールの retain 済みハンドル。そのまま呼び出し側へ渡す
    std::vector<void*>      imH(n, nullptr);

    try {
        id<MTLCommandBuffer> cmd = nil;
        @autoreleasepool {
            cmd = [ctx.queue commandBuffer];
            for (size_t t = 0; t < n; ++t) {
                const MtlOperand a = toOperand(*jobs[t].A);
                const MtlOperand b = toOperand(*jobs[t].B);
                const uint64_t dim = resultDim(op, a.hdr, b.hdr);
                const NSUInteger outBytes = (NSUInteger)checkedTotal(opName(op), dim) * sizeof(float);

                reH[t] = poolAlloc(ctx.device, outBytes);
                imH[t] = poolAlloc(ctx.device, outBytes);
                outs[t].re   = peekBuffer(reH[t]);
                outs[t].im   = peekBuffer(imH[t]);
                outs[t].hash = allocBufOrThrow(ctx.device, sizeof(uint64_t));
                outs[t].coef = allocBufOrThrow(ctx.device, sizeof(NormValHost));
                jobs[t].dim  = static_cast<uint32_t>(dim);

                encodeOp(op, scr[t], ctx, cmd, a, b, outs[t]);
            }
        }
        // ここが「1 往復」の要：全ジョブをまとめて 1 回だけ提出して待つ。
        waitCmdOrThrow(cmd, "batch");
    } catch (...) {
        for (size_t t = 0; t < n; ++t) { poolFree(reH[t]); poolFree(imH[t]); }
        throw;
    }

    for (size_t t = 0; t < n; ++t) {
        const NormValHost* nv = (const NormValHost*)outs[t].coef.contents;
        settleOutput(op, *jobs[t].A, *jobs[t].B, nv->re, nv->im, nv->zeroRe != 0, nv->zeroIm != 0,
                     reH[t], imH[t], jobs[t].outCoef);
        jobs[t].outRe = reH[t];
        jobs[t].outIm = imH[t];
        jobs[t].outId = *(const int64_t*)outs[t].hash.contents;
    }
}

// === バッチ（全ジョブを 1 回の dispatch で回す）===============================
// Heian.cu の runBatchCommon と同じ構成。threadgroup → ジョブ番号の表（blockToOp）を引いて、
// 6 本のカーネル（演算・ノルム部分和・ノルム集約・正規化・ハッシュ部分和・ハッシュ集約）を
// 1 つのエンコーダに積み、commit 1 回・待ち 1 回で済ませる。
// CUDA Graph に相当する仕組みは Metal に無いので、そこだけは揃えていない。
//
// ジョブ記述子は入出力バッファを gpuAddress で持つ（Metal 3 / macOS 13 以降）。
// 間接参照するバッファは useResource で常駐を宣言する必要がある。
// macOS 13 未満、または threadgroup BATCH_TG を起動できない GPU では従来の runBatchCommonMetal を使う。
//
// ★ Heian.metal の JobDesc / BatchOut と並び・型を一致させること
static constexpr NSUInteger BATCH_TG = 256;   // Heian.metal の BATCH_TG と一致させること

struct JobDescMetal {
    GPUInputHeader hdrA;
    GPUInputHeader hdrB;
    uint32_t edgeOffA;      // エッジ連結バッファ内の先頭（QMDD のとき）
    uint32_t edgeOffB;
    uint32_t dim;           // 結果の一辺
    uint32_t total;         // dim*dim
    uint32_t firstBlock;    // このジョブの先頭 threadgroup 番号
    uint32_t partialOff;    // 部分和バッファ内のオフセット
    uint32_t nBlocks;
    uint32_t _pad0;         // 次の 64 ビットを 8 バイト境界に揃える
    uint64_t inReA, inImA;  // gpuAddress。成分が無ければ 0（シェーダは hasRe/hasIm を見て読まない）
    uint64_t inReB, inImB;
    uint64_t outRe, outIm;
};
static_assert(sizeof(GPUInputHeader) == 28,             "GPUInputHeader の大きさが Heian.metal と合わない");
static_assert(offsetof(JobDescMetal, edgeOffA) == 56,   "JobDescMetal の並びが Heian.metal の JobDesc と合わない");
static_assert(offsetof(JobDescMetal, inReA)    == 88,   "JobDescMetal の並びが Heian.metal の JobDesc と合わない");
static_assert(offsetof(JobDescMetal, outIm)    == 128,  "JobDescMetal の並びが Heian.metal の JobDesc と合わない");
static_assert(sizeof(JobDescMetal) == 136,              "JobDescMetal の大きさが Heian.metal の JobDesc と合わない");

struct BatchOutHost {
    float    coefRe;
    float    coefIm;
    uint32_t zeroRe;        // 実部が全要素ゼロなら 1
    uint32_t zeroIm;
    uint64_t hash;
};
static_assert(offsetof(BatchOutHost, hash) == 16, "BatchOutHost の並びが Heian.metal の BatchOut と合わない");
static_assert(sizeof(BatchOutHost) == 24,          "BatchOutHost の大きさが Heian.metal の BatchOut と合わない");

static id<MTLBuffer> newBufWithBytesOrThrow(id<MTLDevice> dev, const void* src, NSUInteger len) {
    id<MTLBuffer> b = [dev newBufferWithBytes:src length:len options:MTLResourceStorageModeShared];
    if (b == nil) {
        throw std::runtime_error("GPU out of memory: " + std::to_string((size_t)len) + " bytes required (batch)");
    }
    return b;
}

API_AVAILABLE(macos(13.0))
static void runBatchFusedMetal(BatchOpKind opKind, BatchJob* jobs, int nJobs) {
    if (!jobs || nJobs <= 0) return;
    auto& ctx = getMetalContext();
    id<MTLDevice> dev = ctx.device;

    std::vector<JobDescMetal>  desc(static_cast<size_t>(nJobs));
    std::vector<uint32_t>      blockToOp;
    std::vector<GPUEdge>       edgeAll;   // 全ジョブのエッジを連結（Heian.cu と同じ）
    std::vector<void*>         outReH(static_cast<size_t>(nJobs), nullptr);
    std::vector<void*>         outImH(static_cast<size_t>(nJobs), nullptr);
    std::vector<id<MTLBuffer>> resident;  // gpuAddress で間接参照するバッファ（useResource 用）
    resident.reserve(static_cast<size_t>(nJobs) * 6);

    auto addrOf = [&](void* h) -> uint64_t {
        if (!h) return 0;                 // 全ゼロ成分はバッファを持たない
        id<MTLBuffer> b = peekBuffer(h);
        resident.push_back(b);
        return static_cast<uint64_t>(b.gpuAddress);
    };

    auto freeOutputs = [&] {
        for (int i = 0; i < nJobs; ++i) { poolFree(outReH[i]); poolFree(outImH[i]); }
    };
    id<MTLBuffer> outBuf = nil;           // 結果（try の外で読む）
    try {
        uint32_t partialAcc = 0;
        for (int i = 0; i < nJobs; ++i) {
            const GPUInput& A = *jobs[i].A;
            const GPUInput& B = *jobs[i].B;
            const GPUInputHeader hA = makeHeader(A);
            const GPUInputHeader hB = makeHeader(B);

            const uint64_t dim64 = resultDim(opKind, hA, hB);
            const uint32_t total = checkedTotal(opName(opKind), dim64);
            const uint32_t dim   = static_cast<uint32_t>(dim64);
            jobs[i].dim = dim;

            const NSUInteger outBytes = static_cast<NSUInteger>(total) * sizeof(float);
            outReH[i] = poolAlloc(dev, outBytes);
            outImH[i] = poolAlloc(dev, outBytes);

            JobDescMetal& d = desc[i];
            std::memset(&d, 0, sizeof(d));
            d.hdrA = hA;
            d.hdrB = hB;
            if (A.kind == SonKind::ArrayLeaf) { d.inReA = addrOf(A.sv.reHandle); d.inImA = addrOf(A.sv.imHandle); }
            if (B.kind == SonKind::ArrayLeaf) { d.inReB = addrOf(B.sv.reHandle); d.inImB = addrOf(B.sv.imHandle); }
            if (A.kind == SonKind::QMDDNode && !A.qmdd.edges.empty()) {
                d.edgeOffA = static_cast<uint32_t>(edgeAll.size());
                edgeAll.insert(edgeAll.end(), A.qmdd.edges.begin(), A.qmdd.edges.end());
            }
            if (B.kind == SonKind::QMDDNode && !B.qmdd.edges.empty()) {
                d.edgeOffB = static_cast<uint32_t>(edgeAll.size());
                edgeAll.insert(edgeAll.end(), B.qmdd.edges.begin(), B.qmdd.edges.end());
            }
            d.outRe = addrOf(outReH[i]);
            d.outIm = addrOf(outImH[i]);
            d.dim   = dim;
            d.total = total;

            const uint32_t g = (total + static_cast<uint32_t>(BATCH_TG) - 1u) / static_cast<uint32_t>(BATCH_TG);
            d.firstBlock = static_cast<uint32_t>(blockToOp.size());
            d.partialOff = partialAcc;
            d.nBlocks    = g;
            blockToOp.insert(blockToOp.end(), g, static_cast<uint32_t>(i));
            partialAcc += g;
        }

        const uint32_t gridTotal = static_cast<uint32_t>(blockToOp.size());
        const uint32_t nJobs32   = static_cast<uint32_t>(nJobs);
        const uint32_t opK       = static_cast<uint32_t>(opKind);
        if (edgeAll.empty()) edgeAll.push_back(GPUEdge{});   // nil を束ねないための詰め物

        id<MTLBuffer> descBuf = newBufWithBytesOrThrow(dev, desc.data(),      desc.size()      * sizeof(JobDescMetal));
        id<MTLBuffer> b2oBuf  = newBufWithBytesOrThrow(dev, blockToOp.data(), blockToOp.size() * sizeof(uint32_t));
        id<MTLBuffer> edgeBuf = newBufWithBytesOrThrow(dev, edgeAll.data(),   edgeAll.size()   * sizeof(GPUEdge));
        const NSUInteger nPart = partialAcc ? partialAcc : 1u;
        id<MTLBuffer> partF   = allocBufOrThrow(dev, nPart * sizeof(float));      // ノルム部分和
        id<MTLBuffer> partU   = allocBufOrThrow(dev, nPart * sizeof(uint64_t));   // ハッシュ部分和
        outBuf = allocBufOrThrow(dev, static_cast<NSUInteger>(nJobs) * sizeof(BatchOutHost));

        id<MTLComputePipelineState> pOp = getPSO(dev, ctx.library, @"batch_op");
        id<MTLComputePipelineState> pSq = getPSO(dev, ctx.library, @"batch_sumsq");
        id<MTLComputePipelineState> pNf = getPSO(dev, ctx.library, @"batch_norm_finalize");
        id<MTLComputePipelineState> pNa = getPSO(dev, ctx.library, @"batch_norm_apply");
        id<MTLComputePipelineState> pH1 = getPSO(dev, ctx.library, @"batch_hash_pass1");
        id<MTLComputePipelineState> pH2 = getPSO(dev, ctx.library, @"batch_hash_finalize");

        const MTLSize tgs    = MTLSizeMake(BATCH_TG, 1, 1);
        const MTLSize gridTG = MTLSizeMake(gridTotal, 1, 1);
        const MTLSize perJob = MTLSizeMake(nJobs32, 1, 1);
        auto perJobTG = [&](id<MTLComputePipelineState> p) {
            NSUInteger n = p.maxTotalThreadsPerThreadgroup;
            if (n > nJobs32) n = nJobs32;
            return MTLSizeMake(n, 1, 1);
        };

        id<MTLCommandBuffer> cmd = nil;
        @autoreleasepool {                     // yield をまたがないようエンコードだけを囲む
            cmd = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];   // 逐次実行のエンコーダ
            for (id<MTLBuffer> b : resident) {
                [enc useResource:b usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
            }

            // 1. 演算本体
            [enc setComputePipelineState:pOp];
            [enc setBuffer:descBuf offset:0 atIndex:0];
            [enc setBuffer:b2oBuf  offset:0 atIndex:1];
            [enc setBuffer:edgeBuf offset:0 atIndex:2];
            [enc setBytes:&opK length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreadgroups:gridTG threadsPerThreadgroup:tgs];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 2. ノルム部分和
            [enc setComputePipelineState:pSq];
            [enc setBuffer:descBuf offset:0 atIndex:0];
            [enc setBuffer:b2oBuf  offset:0 atIndex:1];
            [enc setBuffer:partF   offset:0 atIndex:2];
            [enc dispatchThreadgroups:gridTG threadsPerThreadgroup:tgs];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 3. ノルム集約（1 スレッド = 1 ジョブ）
            [enc setComputePipelineState:pNf];
            [enc setBuffer:descBuf offset:0 atIndex:0];
            [enc setBuffer:partF   offset:0 atIndex:1];
            [enc setBuffer:outBuf  offset:0 atIndex:2];
            [enc setBytes:&nJobs32 length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreads:perJob threadsPerThreadgroup:perJobTG(pNf)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 4. 正規化
            [enc setComputePipelineState:pNa];
            [enc setBuffer:descBuf offset:0 atIndex:0];
            [enc setBuffer:b2oBuf  offset:0 atIndex:1];
            [enc setBuffer:outBuf  offset:0 atIndex:2];
            [enc dispatchThreadgroups:gridTG threadsPerThreadgroup:tgs];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 5. ハッシュ部分和（正規化「後」の値で取る）
            [enc setComputePipelineState:pH1];
            [enc setBuffer:descBuf offset:0 atIndex:0];
            [enc setBuffer:b2oBuf  offset:0 atIndex:1];
            [enc setBuffer:partU   offset:0 atIndex:2];
            [enc setBuffer:outBuf  offset:0 atIndex:3];
            [enc dispatchThreadgroups:gridTG threadsPerThreadgroup:tgs];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // 6. ハッシュ集約（1 スレッド = 1 ジョブ）
            [enc setComputePipelineState:pH2];
            [enc setBuffer:descBuf offset:0 atIndex:0];
            [enc setBuffer:partU   offset:0 atIndex:1];
            [enc setBuffer:outBuf  offset:0 atIndex:2];
            [enc setBytes:&nJobs32 length:sizeof(uint32_t) atIndex:3];
            [enc dispatchThreads:perJob threadsPerThreadgroup:perJobTG(pH2)];
            [enc endEncoding];
        }

        waitCmdOrThrow(cmd, "batch");          // ← 同期はここ 1 回だけ（譲りながら待つ）
    } catch (...) {
        freeOutputs();
        throw;
    }

    const BatchOutHost* res = static_cast<const BatchOutHost*>(outBuf.contents);
    for (int i = 0; i < nJobs; ++i) {
        const BatchOutHost& o = res[i];
        const bool zr = (o.zeroRe != 0), zi = (o.zeroIm != 0);
        settleOutput(opKind, *jobs[i].A, *jobs[i].B, o.coefRe, o.coefIm, zr, zi,
                     outReH[i], outImH[i], jobs[i].outCoef);
        jobs[i].outRe = outReH[i];
        jobs[i].outIm = outImH[i];
        jobs[i].outId = (zr && zi) ? 0 : static_cast<int64_t>(o.hash);
    }
}

// バッチの添字計算は「1 threadgroup = BATCH_TG スレッド」を前提にしている（ホストが
// ceil(total / BATCH_TG) でブロック数を決める）。batch_op はループ内で evalDD を呼ぶので
// レジスタ圧が高く、GPU によっては maxTotalThreadsPerThreadgroup が BATCH_TG を下回りうる。
// そうなると添字がずれて黙って壊れるので、6 本とも BATCH_TG で起動できるかを一度だけ確かめ、
// 足りなければ従来経路を使う。
static bool fusedBatchSupported() {
    static const bool v = [] {
        if (@available(macOS 13.0, *)) {
            auto& ctx = getMetalContext();
            for (NSString* fn in @[@"batch_op", @"batch_sumsq", @"batch_norm_finalize",
                                    @"batch_norm_apply", @"batch_hash_pass1", @"batch_hash_finalize"]) {
                id<MTLComputePipelineState> p = getPSO(ctx.device, ctx.library, fn);
                if (p == nil || p.maxTotalThreadsPerThreadgroup < BATCH_TG) {
                    fprintf(stderr, "[Heian.mm] %s は threadgroup %zu を起動できないため従来のバッチを使います\n",
                            fn.UTF8String, (size_t)BATCH_TG);
                    return false;
                }
            }
            return true;
        }
        return false;
    }();
    return v;
}

static void dispatchBatch(BatchOpKind opKind, BatchJob* jobs, int n) {
    if (fusedBatchSupported()) {
        if (@available(macOS 13.0, *)) { runBatchFusedMetal(opKind, jobs, n); return; }
    }
    runBatchCommonMetal(opKind, jobs, n);
}

extern "C" void runMulBatchWrapper(BatchJob* jobs, int n)  { dispatchBatch(BATCH_OP_MUL,  jobs, n); }
extern "C" void runAddBatchWrapper(BatchJob* jobs, int n)  { dispatchBatch(BATCH_OP_ADD,  jobs, n); }
extern "C" void runKronBatchWrapper(BatchJob* jobs, int n) { dispatchBatch(BATCH_OP_KRON, jobs, n); }

extern "C" bool copyGpuBufferToHostFloat(void* gpuHandle, float* dst, size_t count) {
    if (!gpuHandle || !dst) return false;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)gpuHandle;
    if (!buf.contents) return false;
    if (buf.length < count * sizeof(gpu_precision)) return false;
    std::memcpy(dst, buf.contents, count * sizeof(float));
    return true;
}

// 呼び出し側にはバックエンドの精度を見せない：Metal のバッファは
// gpu_precision (= float) なので、ここで double へ広げて返す。
extern "C" bool copyGpuBufferToHostDouble(void* gpuHandle, double* dst, size_t count) {
    if (!gpuHandle || !dst) return false;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)gpuHandle;
    if (!buf.contents) return false;
    if (buf.length < count * sizeof(gpu_precision)) return false;
    const gpu_precision* src = (const gpu_precision*)buf.contents;
    for (size_t k = 0; k < count; ++k) dst[k] = static_cast<double>(src[k]);
    return true;
}


// ArrayLeaf のデストラクタから呼ばれる。解放せずプールへ返す（CUDA 側 Heian.cu と同じ）。
void releaseGpuBuffer(void* p) {
    poolFree(p);
}