#include "mathUtils.hpp"
#include <vector>
#include <atomic>

// 閾値の 1 つ上のノードでは、独立な子演算をまとめて 1 往復で処理する。
// 桁合わせは実体化せず、倍率だけ渡してカーネル側で単位行列拡張として読ませる。

// 密葉（ArrayLeaf）の一意 id。内容に依らず単調増加カウンタで振る（重複排除はしない）。
//
// 背景: GPU の内容ハッシュを id にすると、量子状態のような構造的データで系統的に
// 衝突し（portfolio で約2%）、その id をキーにした演算キャッシュ/重複排除が
// 別内容の葉を取り違えて状態を壊した。内容ハッシュを collision-free にするのは
// GPU 並列ハッシュでは難しいため、密葉は必ず一意の id を持たせ、id による葉の
// 同一視（重複排除）を構造的に起こさない設計にする。
//
// id 空間は他と分離する: 終端=0、QMDDNode の key=正の63bit（XXH3 & 0x7fff...）。
// 密葉 id は最上位ビットを立てて「負値」にし、ノードキー・終端と絶対に重ならない。
// atomic なので複数スレッド/ファイバーから同時に呼んでも重複しない。
static int64_t arrayLeafId() {
    static std::atomic<uint64_t> ctr{1};
    return static_cast<int64_t>(ctr.fetch_add(1, std::memory_order_relaxed) | 0x8000000000000000ULL);
}


// 行優先で最初の非ゼロを係数として括り出す正規化。
// GPU 側（Heian.cu / Heian.metal）の正規化もこの規約に合わせてあること。
static complex<double> normalizePivot(EdgeMatrix& z, bool& allWeightsAreZero) {
    complex<double> tmpWeight = .0;
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            if (mathUtils::isZERO(z[i][j].weight)) continue;
            allWeightsAreZero = false;
            if (mathUtils::isZERO(tmpWeight)) {
                tmpWeight = z[i][j].weight;
                z[i][j].weight = 1.0;
            } else {
                z[i][j].weight /= tmpWeight;
            }
        }
    }
    return tmpWeight;
}

// 閾値以下（または密葉が絡む）の mul を GPU で 1 回実行する。
static QMDDEdge mulDispatch(const QMDDEdge& a, const QMDDEdge& b) {
    const int goalDepth = max(a.level(), b.level());
    
    // 案2: 桁合わせを実体化しない。倍率だけ渡してカーネル側で A ⊗ I として読む。
    GPUInput A = farewell(a);  A.pad = 1u << (goalDepth - a.level());
    GPUInput B = farewell(b);  B.pad = 1u << (goalDepth - b.level());

    void* outRe = nullptr;
    void* outIm = nullptr;

    int64_t outId = 0;
    gpu_precision outCoef[2] = {.0f, .0f};

    runMulAny2Wrapper(A, B,
                    &outRe, &outIm,
                    &outId, outCoef);
    if (outCoef[0] == .0f && outCoef[1] == .0f) {
        releaseGpuBuffer(outRe);
        releaseGpuBuffer(outIm);
        return edgeZero;
    }
    { int64_t lid = arrayLeafId(); return QMDDEdge(complex<double>(outCoef[0], outCoef[1]), lid, pluck(A.dim * A.pad, outRe, outIm, lid)); }
}

// 閾値の 1 つ上のノードの mul。(i,j,k) の 8 つの掛け算をまとめて 1 往復で GPU に投げ、
// そのあとの足し合わせ（k について直列）は従来どおり add に任せる。
static EdgeMatrix mulBatchDispatch(const QMDDEdge& a, const QMDDEdge& b, int depth) {
    QMDDNode* n0 = a.node();
    QMDDNode* n1 = b.node();
    EdgeMatrix z{{ {edgeZero, edgeZero}, {edgeZero, edgeZero} }};

    struct Slot { size_t i, j, k; };
    std::vector<Slot>     slots; slots.reserve(8);
    std::vector<GPUInput> insA;  insA.reserve(8);
    std::vector<GPUInput> insB;  insB.reserve(8);
    std::vector<BatchJob> jobs;  jobs.reserve(8);
    // GPUInput は生ポインタしか持たない。ループ内で作った p / q を
    // ここで保持しておかないと、バッチ実行前に ArrayLeaf が破棄されて
    // バッファがプールに返り、出力用に再確保されてしまう。
    std::vector<QMDDEdge> keepA; keepA.reserve(8);
    std::vector<QMDDEdge> keepB; keepB.reserve(8);
    QMDDEdge part[2][2][2];                       // [i][j][k] の掛け算結果
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            for (size_t k = 0; k < 2; k++) part[i][j][k] = edgeZero;

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                QMDDEdge p = n0->edges[i][k];   // 既存エッジのコピー（キー・種別・子をそのまま。再ハッシュしない）
                QMDDEdge q = n1->edges[k][j];
                if (q.isTerminal) std::swap(p, q);
                if (p.isTerminal) {               // 従来の早期 return と同じ扱い
                    if (p.weight == .0)       part[i][j][k] = p;
                    else if (p.weight == 1.0) part[i][j][k] = q;
                    else                      { QMDDEdge r = q; r.weight = p.weight * q.weight; part[i][j][k] = r; }
                    continue;
                }
                const int goal = max(p.level(), q.level());
                slots.push_back({i, j, k});
                keepA.push_back(p);
                keepB.push_back(q);
                insA.push_back(farewell(p));
                insB.push_back(farewell(q));
                insA.back().pad = 1u << (goal - p.level());
                insB.back().pad = 1u << (goal - q.level());
            }
        }
    }
    for (size_t t = 0; t < insA.size(); ++t) {
        BatchJob jb; jb.A = &insA[t]; jb.B = &insB[t];
        jobs.push_back(jb);
    }
    if (!jobs.empty()) {
        runMulBatchWrapper(jobs.data(), static_cast<int>(jobs.size()));
        for (size_t t = 0; t < jobs.size(); ++t) {
            const BatchJob& jb = jobs[t];
            if (jb.outCoef[0] == .0f && jb.outCoef[1] == .0f) {
                releaseGpuBuffer(jb.outRe);
                releaseGpuBuffer(jb.outIm);
                part[slots[t].i][slots[t].j][slots[t].k] = edgeZero;
            } else {
                part[slots[t].i][slots[t].j][slots[t].k] =
                    [&]{ int64_t lid = arrayLeafId(); return QMDDEdge(complex<double>(jb.outCoef[0], jb.outCoef[1]), lid, pluck(jb.dim, jb.outRe, jb.outIm, lid)); }();
            }
        }
    }
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 2; j++)
            z[i][j] = mathUtils::add(mathUtils::add(edgeZero, part[i][j][0], depth + 1),
                                     part[i][j][1], depth + 1);
    return z;
}

QMDDEdge mathUtils::mul(const QMDDEdge& e0, const QMDDEdge& e1, int depth) {
    QMDDEdge a = e0;
    QMDDEdge b = e1;
    if (b.isTerminal) std::swap(a, b);
    if (a.sonKind_ == SonKind::Terminal) {
        if (a.weight == .0) {
            return a;
        } else if (a.weight == 1.0){
            return b;
        } else {
            { QMDDEdge r = b; r.weight = a.weight * b.weight; return r; }
        }
    }
    if (depth >= PARAMETER.parallelism.GPU || a.sonKind_ == SonKind::ArrayLeaf || b.sonKind_ == SonKind::ArrayLeaf) {
        return mulDispatch(a, b);
    }

    bool concurrency = depth < PARAMETER.parallelism.fiber;

    int64_t operationCacheKey = calculation::generateOperationCacheKey(OperationKey(a.key_, b.key_));
    OperationCacheClient& cache = OperationCacheClient::getInstance();
    // 深さ 0（ゲート × 状態 の最上位）はキャッシュしない: 同じ組が再登場することはなく、
    // かつ最上位の呼び出しはシミュレーション全体を通して生きる fiber で走るので、その局所キャッシュに
    // 各ステップの状態 DD 全体（密葉の GPU バッファ込み）が溜まり続けてしまう（2026-09-12 に GPU メモリ枯渇で確認）。
    // 深さ 1 以上は演算ごとの fiber の局所キャッシュで、fiber の終了とともに消える。
    const bool useCache = PARAMETER.cache.alive && depth > 0;
    if (useCache) {
        // 演算キャッシュは常に fiber ごとの局所側を使う。グローバル側（GraalVM）は 1 参照が数十 µs かかる。
        if (auto existingEdge = cache.find(operationCacheKey, /*useLocal=*/true)) {
            if (existingEdge->weight != .0 && existingEdge->key_ != 0) {
                QMDDEdge result = *existingEdge;   // キャッシュのエッジをコピー（再登録なし）
                result.weight *= a.weight * b.weight;
                return result;
            }
        }
    }


    QMDDNode* n0 = a.node();
    QMDDNode* n1 = b.node();

    EdgeMatrix z{{ {edgeZero, edgeZero}, {edgeZero, edgeZero} }};
    

    vector<boost::fibers::future<pair<pair<size_t, size_t>, QMDDEdge>>> fiberFutures;


    if (depth + 1 == PARAMETER.parallelism.GPU) {
        z = mulBatchDispatch(a, b, depth);
    } else
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            array<int, 4> depths{
                n0->edges[i][0].level(),
                n0->edges[i][1].level(),
                n1->edges[0][j].level(),
                n1->edges[1][j].level()
            };
            double calculatedDepth = mathUtils::median(depths);
            if (concurrency) {
                fiberFutures.emplace_back(
                    boost::fibers::async([&, i, j]() -> pair<pair<size_t, size_t>, QMDDEdge> {
                        QMDDEdge answer = edgeZero;
                        for (size_t k = 0; k < 2; k++) {
                            QMDDEdge p = n0->edges[i][k];   // 既存エッジのコピー（キー・種別・子をそのまま。再ハッシュしない）
                            QMDDEdge q = n1->edges[k][j];
                            answer = mathUtils::add(answer, mathUtils::mul(p, q, depth + 1), depth + 1);
                        }
                        return {{i, j}, answer};
                    })
                );
            } else {
                QMDDEdge answer = edgeZero;
                for (size_t k = 0; k < 2; k++) {
                    QMDDEdge p = n0->edges[i][k];   // 既存エッジのコピー（キー・種別・子をそのまま。再ハッシュしない）
                    QMDDEdge q = n1->edges[k][j];
                    answer = mathUtils::add(answer, mathUtils::mul(p, q, depth + 1), depth + 1);
                }
                z[i][j] = answer;
            }
        }
    }


    bool allWeightsAreZero = true;
    for (auto& ff : fiberFutures) {
        const auto& [indices, result] = ff.get();
        const auto& [i, j] = indices;
        z[i][j] = result;
    }

    // 正規化：行優先で最初の非ゼロ要素を係数として取り出し、その要素を 1 に、
    // 残りをその係数で割る。ゼロ要素は 0/c = 0 なのでそのまま。
    complex<double> tmpWeight = normalizePivot(z, allWeightsAreZero);

    // ノードの生成・登録は 1 回だけ。キャッシュ用と結果用で同じノードを共有する
    // （以前は同じ z から 2 個作っていたので、cache ON だと登録が 2 回走っていた）。
    if (allWeightsAreZero && !useCache) return edgeZero;
    QMDDEdge normalized(tmpWeight, bet(z));
    if (useCache) {
        cache.insert(operationCacheKey, normalized, /*useLocal=*/true);
    }
    if (allWeightsAreZero) return edgeZero;
    QMDDEdge result = normalized;                       // コピー（再登録なし）。重みは tmpWeight
    result.weight *= a.weight * b.weight;
    return result;
}

// QMDDEdge mathUtils::mulForDiagonal(const QMDDEdge& a, const QMDDEdge& b) {

//     jniUtils& cache = jniUtils::getInstance();
//     int64_t operationCacheKey = calculation::generateOperationCacheKey(
//         OperationKey(a, OperationType::MUL, b)
//     );

//     auto cacheFuture = threadPool.enqueue([&cache, operationCacheKey]() -> QMDDEdge {
//         OperationResult existing = cache.jniFind(operationCacheKey);
//         if (existing != OperationResult{.0, 0}) {
//             QMDDEdge answer{ existing.first, existing.second };
//             if (answer.key_!= 0) {
//                 return answer;
//             }
//         }
//         return edgeZero;
//     });

//     auto computeFuture = threadPool.enqueue([=]() -> QMDDEdge {
//         if (b.isTerminal) std::swap(const_cast<QMDDEdge&>(a), const_cast<QMDDEdge&>(b));
//         if (a.isTerminal) {
//             if (a.weight == .0)         return a;
//             else if (a.weight == 1.0)   return b;
//             else                         return QMDDEdge(a.weight * b.weight, b.key_);
//         }
//         auto n0 = a.getSon();
//         auto n1 = b.getSon();
//         vector<vector<QMDDEdge>> z(2, std::vector<QMDDEdge>(2, edgeZero));
//         complex<double> tmpWeight = .0;
//         bool allZero = true;

//         for (size_t n = 0; n < 2; n++) {
//             QMDDEdge p(a.weight * n0->edges[n][n].weight, n0->edges[n][n].key_);
//             QMDDEdge q(b.weight * n1->edges[n][n].weight, n1->edges[n][n].key_);
//             z[n][n] = mathUtils::mulForDiagonal(p, q);
//             if (z[n][n].weight != .0) {
//                 allZero = false;
//                 if (tmpWeight == .0) {
//                     tmpWeight = z[n][n].weight;
//                     z[n][n].weight = 1.0;
//                 } else {
//                     z[n][n].weight /= tmpWeight;
//                 }
//             }
//         }

//         QMDDEdge result;
//         if (allZero) {
//             result = edgeZero;
//         } else {
//             result = QMDDEdge(a.weight * tmpWeight, bet(z));
//         }
//         return result;
//     });


//     while (true) {
//         if (cacheFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge cached = cacheFuture.get();
//             if (cached != edgeZero) {
//                 return cached;
//             }

//             QMDDEdge computed = computeFuture.get();

//             threadPool.enqueue([&cache, operationCacheKey, computed]() {
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             });
//             return computed;
//         }

//         if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge computed = computeFuture.get();
//             threadPool.enqueue([&cache, operationCacheKey, computed]() {
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             });
//             return computed;
//         }
//     }
// }

// 閾値以下（または密葉が絡む）の add を GPU で 1 回実行する。
static QMDDEdge addDispatch(const QMDDEdge& a, const QMDDEdge& b) {
    const int goalDepth = max(a.level(), b.level());
    
    // 案2: 桁合わせを実体化しない。倍率だけ渡してカーネル側で A ⊗ I として読む。
    GPUInput A = farewell(a);  A.pad = 1u << (goalDepth - a.level());
    GPUInput B = farewell(b);  B.pad = 1u << (goalDepth - b.level());

    void* outRe = nullptr;
    void* outIm = nullptr;

    int64_t outId = 0;
    gpu_precision outCoef[2] = {.0f, .0f};

    runAddAny2Wrapper(A, B,
                    &outRe, &outIm,
                    &outId, outCoef);
    if (outCoef[0] == .0f && outCoef[1] == .0f) {
        releaseGpuBuffer(outRe);
        releaseGpuBuffer(outIm);
        return edgeZero;
    }
    { int64_t lid = arrayLeafId(); return QMDDEdge(complex<double>(outCoef[0], outCoef[1]), lid, pluck(A.dim * A.pad, outRe, outIm, lid)); }
}

// 閾値の 1 つ上のノードの add。4 マスは互いに独立なので、短絡できるものを先に片付け、
// 残りをまとめて 1 往復で GPU に投げる。
static EdgeMatrix addBatchDispatch(const QMDDEdge& a, const QMDDEdge& b) {
    QMDDNode* n0 = a.node();
    QMDDNode* n1 = b.node();
    EdgeMatrix z{{ {edgeZero, edgeZero}, {edgeZero, edgeZero} }};

    struct Slot { size_t i, j; };
    std::vector<Slot>     slots; slots.reserve(4);
    std::vector<GPUInput> insA;  insA.reserve(4);
    std::vector<GPUInput> insB;  insB.reserve(4);
    std::vector<BatchJob> jobs;  jobs.reserve(4);
    // ループ内で作ったオペランドをバッチ実行まで生かしておく（mul と同じ理由）
    std::vector<QMDDEdge> keepA; keepA.reserve(4);
    std::vector<QMDDEdge> keepB; keepB.reserve(4);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            QMDDEdge p = n0->edges[i][j]; p.weight *= a.weight;   // コピーして重みだけ掛ける
            QMDDEdge q = n1->edges[i][j]; q.weight *= b.weight;
            if (q.isTerminal) std::swap(p, q);
            if (p.isTerminal) {                       // 従来の早期 return と同じ扱い
                z[i][j] = q.isTerminal ? QMDDEdge(p.weight + q.weight) : q;
                continue;
            }
            // add ショートカット削除（メイン経路と同じ理由）: key 一致でも内容一致を
            // 保証できないため、重みの和で済ませず通常の加算に回す。
            const int goal = max(p.level(), q.level());
            slots.push_back({i, j});
            keepA.push_back(p);
            keepB.push_back(q);
            insA.push_back(farewell(p));
            insB.push_back(farewell(q));
            insA.back().pad = 1u << (goal - p.level());
            insB.back().pad = 1u << (goal - q.level());
        }
    }
    for (size_t k = 0; k < insA.size(); ++k) {
        BatchJob jb; jb.A = &insA[k]; jb.B = &insB[k];
        jobs.push_back(jb);
    }
    if (!jobs.empty()) {
        runAddBatchWrapper(jobs.data(), static_cast<int>(jobs.size()));
        for (size_t k = 0; k < jobs.size(); ++k) {
            const BatchJob& jb = jobs[k];
            if (jb.outCoef[0] == .0f && jb.outCoef[1] == .0f) {
                releaseGpuBuffer(jb.outRe);
                releaseGpuBuffer(jb.outIm);
                z[slots[k].i][slots[k].j] = edgeZero;
            } else {
                z[slots[k].i][slots[k].j] =
                    [&]{ int64_t lid = arrayLeafId(); return QMDDEdge(complex<double>(jb.outCoef[0], jb.outCoef[1]), lid, pluck(jb.dim, jb.outRe, jb.outIm, lid)); }();
            }
        }
    }
    return z;
}

QMDDEdge mathUtils::add(const QMDDEdge& e0, const QMDDEdge& e1, int depth) {
    QMDDEdge a = e0;
    QMDDEdge b = e1;
    if (b.isTerminal) std::swap(a, b);
    if (a.isTerminal) {
        if (b.isTerminal) {
            return QMDDEdge(a.weight + b.weight);
        }
        // a が終端（ゼロ/スカラ）で b がノード：次元が合わないので b を返す（fall-through で get<QMDDNode>(終端) を踏むのを防ぐ）
        return b;
    }
    // add ショートカット削除: 子の key が同じでも、GPU 密葉では id(内容ハッシュ)一致が
    // 内容一致を保証しないケースがあり、重みの和だけで済ませると誤マージになる。
    // 常に通常の加算経路（下の GPU / 再帰）に回す。
    if (depth >= PARAMETER.parallelism.GPU || a.sonKind_ == SonKind::ArrayLeaf || b.sonKind_ == SonKind::ArrayLeaf) {
        return addDispatch(a, b);
    }

    bool concurrency = depth < PARAMETER.parallelism.fiber;

    QMDDNode* n0 = a.node();
    QMDDNode* n1 = b.node();
    EdgeMatrix z{{ {edgeZero, edgeZero}, {edgeZero, edgeZero} }};


    vector<boost::fibers::future<pair<pair<size_t, size_t>, QMDDEdge>>> fiberFutures;

    if (depth + 1 == PARAMETER.parallelism.GPU) {
        z = addBatchDispatch(a, b);
    } else
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            array<int, 2> depths{
                n0->edges[i][j].level(),
                n1->edges[i][j].level()
            };
            if (concurrency) {
                fiberFutures.emplace_back(
                    boost::fibers::async([&, i, j]() -> pair<pair<size_t, size_t>, QMDDEdge> {
                        QMDDEdge p = n0->edges[i][j]; p.weight *= a.weight;   // コピーして重みだけ掛ける
                        QMDDEdge q = n1->edges[i][j]; q.weight *= b.weight;
                        QMDDEdge r = mathUtils::add(p, q, depth + 1);
                        return {{i, j}, r};
                    })
                );
            } else {
                QMDDEdge p = n0->edges[i][j]; p.weight *= a.weight;   // コピーして重みだけ掛ける
                QMDDEdge q = n1->edges[i][j]; q.weight *= b.weight;
                z[i][j] = mathUtils::add(p, q, depth + 1);
            }
        }
    }

    for (auto& ff : fiberFutures) {
        const auto& [indices, result] = ff.get();
        const auto& [i, j] = indices;
        z[i][j] = result;
    }

    bool allWeightsAreZero = true;

    // 正規化：行優先で最初の非ゼロ要素を係数として取り出し、その要素を 1 に、
    // 残りをその係数で割る。ゼロ要素は 0/c = 0 なのでそのまま。
    complex<double> tmpWeight = normalizePivot(z, allWeightsAreZero);

    QMDDEdge result = allWeightsAreZero ? edgeZero : QMDDEdge(tmpWeight, bet(z));
    return result;
}

// QMDDEdge mathUtils::addForDiagonal(const QMDDEdge& a, const QMDDEdge& b) {
//     jniUtils& cache = jniUtils::getInstance();
//     int64_t operationCacheKey = calculation::generateOperationCacheKey(
//         OperationKey(a, OperationType::ADD, b)
//     );

//     auto cacheFuture = threadPool.enqueue([&cache, operationCacheKey]() -> QMDDEdge {
//         OperationResult existing = cache.jniFind(operationCacheKey);
//         if (existing != OperationResult{.0, 0}) {
//             QMDDEdge answer{ existing.first, existing.second };
//             if (answer.key_ != 0) {
//                 return answer;
//             }
//         }
//         return edgeZero;
//     });

//     auto computeFuture = threadPool.enqueue([=]() -> QMDDEdge {
//         if (b.isTerminal) std::swap(const_cast<QMDDEdge&>(a), const_cast<QMDDEdge&>(b));
//         if (a.isTerminal) {
//             if (a.weight == .0)         return b;
//             else if (b.isTerminal)      return QMDDEdge(a.weight + b.weight, 0);
//         }
//         auto n0 = a.getSOn();
//         auto n1 = b.getSon();
//         EdgeMatrix z{{ {edgeZero, edgeZero}, {edgeZero, edgeZero} }};
//         complex<double> tmpWeight = .0;
//         bool allZero = true;

//         for (size_t n = 0; n < 2; n++) {
//             QMDDEdge p(a.weight * n0->edges[n][n].weight, n0->edges[n][n].key_);
//             QMDDEdge q(b.weight * n1->edges[n][n].weight, n1->edges[n][n].key_);
//             z[n][n] = mathUtils::addForDiagonal(n0->edges[n][n], b);
//             if (z[n][n].weight != .0) {
//                 allZero = false;
//                 if (tmpWeight == .0) {
//                     tmpWeight = z[n][n].weight;
//                     z[n][n].weight = 1.0;
//                 } else {
//                     z[n][n].weight /= tmpWeight;
//                 }
//             }
//         }

//         QMDDEdge result;
//         if (allZero) {
//             result = edgeZero;
//         } else {
//             result = QMDDEdge(a.weight * tmpWeight, bet(z));
//         }
//         return result;
//     });


//     while (true) {
//         if (cacheFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge cached = cacheFuture.get();
//             if (cached != edgeZero) {
//                 return cached;
//             }

//             QMDDEdge computed = computeFuture.get();

//             threadPool.enqueue([&cache, operationCacheKey, computed]() {
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             });
//             return computed;
//         }

//         if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge computed = computeFuture.get();
//             threadPool.enqueue([&cache, operationCacheKey, computed]() {
//                 cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             });
//             return computed;
//         }
//     }
// }

// 閾値以下（または密葉が絡む）の kron を GPU で 1 回実行する。
static QMDDEdge kronDispatch(const QMDDEdge& e0, const QMDDEdge& e1) {
    GPUInput A = farewell(e0);
    GPUInput B = farewell(e1);

    void* outRe = nullptr;
    void* outIm = nullptr;
    int64_t outId = 0;
    gpu_precision outCoef[2] = {.0f, .0f};

    runKronAny2Wrapper(A, B,
                    &outRe, &outIm,
                    &outId, outCoef);
    if (outCoef[0] == .0f && outCoef[1] == .0f) {
        releaseGpuBuffer(outRe);
        releaseGpuBuffer(outIm);
        return edgeZero;
    }
    { int64_t lid = arrayLeafId(); return QMDDEdge(complex<double>(outCoef[0], outCoef[1]), lid, pluck(A.dim * B.dim, outRe, outIm, lid)); }
}

// 閾値の 1 つ上のノードの kron。子 4 つは互いに独立なので、短絡できるものを先に片付け、
// 残りをまとめて 1 往復で GPU に投げる。
static EdgeMatrix kronBatchDispatch(const QMDDEdge& e0, const QMDDEdge& e1) {
    QMDDNode* n0 = e0.node();
    EdgeMatrix z{};

    struct Slot { size_t i, j; };
    std::vector<Slot>    slots;    slots.reserve(4);
    std::vector<GPUInput> insA;    insA.reserve(4);
    std::vector<BatchJob> jobs;    jobs.reserve(4);
    std::vector<QMDDEdge> keepA;   keepA.reserve(4);   // オペランドをバッチ実行まで保持
    GPUInput inB = farewell(e1);   // e1 は 4 つで共通

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            const QMDDEdge& c = n0->edges[i][j];
            if (c.isTerminal) {                       // 従来の早期 return と同じ扱い
                if (c.weight == .0)       z[i][j] = c;
                else if (c.weight == 1.0) z[i][j] = e1;
                else                      { QMDDEdge r = e1; r.weight = c.weight * e1.weight; z[i][j] = r; }
                continue;
            }
            slots.push_back({i, j});
            keepA.push_back(c);
            insA.push_back(farewell(c));
        }
    }
    for (size_t k = 0; k < insA.size(); ++k) {
        BatchJob jb; jb.A = &insA[k]; jb.B = &inB;
        jobs.push_back(jb);
    }
    if (!jobs.empty()) {
        runKronBatchWrapper(jobs.data(), static_cast<int>(jobs.size()));
        for (size_t k = 0; k < jobs.size(); ++k) {
            const BatchJob& jb = jobs[k];
            QMDDEdge r;
            if (jb.outCoef[0] == .0f && jb.outCoef[1] == .0f) {
                releaseGpuBuffer(jb.outRe);
                releaseGpuBuffer(jb.outIm);
                r = edgeZero;
            } else {
                r = [&]{ int64_t lid = arrayLeafId(); return QMDDEdge(complex<double>(jb.outCoef[0], jb.outCoef[1]), lid, pluck(jb.dim, jb.outRe, jb.outIm, lid)); }();
            }
            z[slots[k].i][slots[k].j] = r;
        }
    }
    return z;
}

QMDDEdge mathUtils::kron(const QMDDEdge& e0, const QMDDEdge& e1, int depth) {
    if (e0.isTerminal) {
        if (e0.weight == .0) {
            return e0;
        }else if (e0.weight == 1.0) {
            return e1;
        } else {
            { QMDDEdge r = e1; r.weight = e0.weight * e1.weight; return r; }
        }
    }

    if (depth >= PARAMETER.parallelism.GPU || e0.sonKind_ == SonKind::ArrayLeaf) {
        return kronDispatch(e0, e1);
    }
    QMDDNode* n0 = e0.node();

    EdgeMatrix z{};

    if (depth + 1 == PARAMETER.parallelism.GPU) {
        z = kronBatchDispatch(e0, e1);
    } else
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            z[i][j] = mathUtils::kron(n0->edges[i][j], e1, depth + 1);

        }
    }

    bool allWeightsAreZero = true;
    complex<double> tmpWeight = normalizePivot(z, allWeightsAreZero);
    QMDDEdge result = allWeightsAreZero ? edgeZero : QMDDEdge(e0.weight * tmpWeight, bet(z));
    return result;
}

// QMDDEdge mathUtils::kron(const QMDDEdge& e0, const QMDDEdge& e1) {
//     if (e0.isTerminal) {
//         if (e0.weight == .0) {
//             return e0;
//         }else if (e0.weight == 1.0) {
//             return e1;
//         } else {
//             return QMDDEdge(e0.weight * e1.weight, e1.key_);
//         }
//     }
//     shared_ptr<QMDDNode> n0 = e0.getSon();
//     shared_ptr<QMDDNode> n1 = e1.getSon();
//     vector<vector<QMDDEdge>> z(n0->edges.size(), vector<QMDDEdge>(n1->edges[0].size()));
//     for (size_t i = 0; i < n0->edges.size(); i++) {
//         for (size_t j = 0; j < n0->edges[i].size(); j++) {
//             z[i][j] = QMDDEdge(n0->edges[i][j].weight, e1.key_);
//         }
//     }

//     QMDDEdge result = QMDDEdge(e0.weight * e1.weight, bet(z));
//     return result;
// }

// QMDDEdge mathUtils::kronForDiagonal(const QMDDEdge& e0, const QMDDEdge& e1) {

//     jniUtils& cache = jniUtils::getInstance();
//     int64_t operationCacheKey = calculation::generateOperationCacheKey(
//         OperationKey(e0, OperationType::KRONECKER, e1)
//     );

//     auto cacheFuture = threadPool.enqueue([&cache, operationCacheKey]() -> QMDDEdge {
//         OperationResult existing = cache.jniFind(operationCacheKey);
//         if (existing != OperationResult{.0, 0}) {
//             QMDDEdge answer{ existing.first, existing.second };
//             if (answer.key_ != 0) {
//                 return answer;
//             }
//         }
//         return edgeZero;
//     });

//     auto computeFuture = threadPool.enqueue([=]() -> QMDDEdge {
//         if (e0.isTerminal) {
//             if (e0.weight == .0)         return e0;
//             else if (e0.weight == 1.0)   return e1;
//             else                         return QMDDEdge(e0.weight * e1.weight, e1.key_);
//         }
//         auto n0 = e0.getSon();
//         auto n1 = e1.getSon();
//         vector<vector<QMDDEdge>> z(2, std::vector<QMDDEdge>(2, edgeZero));
//         complex<double> tmpWeight = .0;
//         bool allZero = true;

//         for (size_t n = 0; n < 2; n++) {
//             z[n][n] = mathUtils::kronForDiagonal(n0->edges[n][n], e1);
//             if (z[n][n].weight != .0) {
//                 allZero = false;
//                 if (tmpWeight == .0) {
//                     tmpWeight = z[n][n].weight;
//                     z[n][n].weight = 1.0;
//                 } else {
//                     z[n][n].weight /= tmpWeight;
//                 }
//             }
//         }

//         QMDDEdge result;
//         if (allZero) {
//             result = edgeZero;
//         } else {
//             result = QMDDEdge(e0.weight * tmpWeight, bet(z));
//         }
//         return result;
//     });


//     while (true) {
//         if (cacheFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge cached = cacheFuture.get();
//             if (cached != edgeZero) {
//                 return cached;
//             }

//             QMDDEdge computed = computeFuture.get();

//             // threadPool.enqueue([&cache, operationCacheKey, computed]() {
//             //     cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             // });
//             return computed;
//         }

//         if (computeFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//             QMDDEdge computed = computeFuture.get();
//             // threadPool.enqueue([&cache, operationCacheKey, computed]() {
//             //     cache.jniInsert(operationCacheKey, computed.weight, computed.key_);
//             // });
//             return computed;
//         }
//     }
// }

QMDDEdge mathUtils::dyad(const QMDDEdge& e0, const QMDDEdge& e1) {
    if (e0.isTerminal || e1.isTerminal) {
        return QMDDEdge(e0.weight * e1.weight);
    }
    QMDDNode* n0 = e0.node();
    QMDDNode* n1 = e1.node();
    EdgeMatrix z{};
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            z[i][j] = mathUtils::dyad(n0->edges[i][0], n1->edges[0][j]);
        }
    }
    QMDDEdge result;
    result = QMDDEdge(1.0, bet(z));
    return result;
}


double mathUtils::csc(const double theta) {
    double sin_theta = sin(theta);
    if (sin_theta == .0) throw overflow_error("csc(θ) is undefined (sin(θ) = 0)");
    return 1.0 / sin_theta;
}

complex<double> mathUtils::csc(const complex<double> theta) {
    complex<double> sin_theta = sin(theta);
    if (sin_theta == .0) throw overflow_error("csc(θ) is undefined (sin(θ) = 0)");
    return 1.0 / sin_theta;
}

double mathUtils::sec(const double theta) {
    double cos_theta = cos(theta);
    if (cos_theta == .0) throw overflow_error("sec(θ) is undefined (cos(θ) = 0)");
    return 1.0 / cos_theta;
}

complex<double> mathUtils::sec(const complex<double> theta) {
    complex<double> cos_theta = cos(theta);
    if (cos_theta == .0) throw overflow_error("sec(θ) is undefined (cos(θ) = 0)");
    return 1.0 / cos_theta;
}

double mathUtils::cot(const double theta) {
    double tan_theta = tan(theta);
    if (tan_theta == .0) throw overflow_error("cot(θ) is undefined (tan(θ) = 0)");
    return 1.0 / tan_theta;
}

complex<double> mathUtils::cot(const complex<double> theta) {
    complex<double> tan_theta = tan(theta);
    if (tan_theta == .0) throw overflow_error("cot(θ) is undefined (tan(θ) = 0)");
    return 1.0 / tan_theta;
}

// double mathUtils::sumOfSquares(const vector<complex<double>>& vec) {
//     return accumulate(vec.begin(), vec.end(), 0.0, [](double sum, const complex<double>& val) {
//         return sum + std::pow(abs(val), 2);
//     });
// }

double mathUtils::sumOfSquares(const vector<complex<double>>& vec) {
    return accumulate(vec.begin(), vec.end(), 0.0, [](double sum, const complex<double>& val) {
        return sum + std::pow(abs(val), 2);
    });
}

vector<int> mathUtils::createRange(int start, int end) {
    int min = std::min(start, end);
    int max = std::max(start, end);
    vector<int> range;

    for (int i = min; i <= max; ++i) {
        range.push_back(i);
    }
    return range;
}

int mathUtils::findCoprimeBelow(int N) {
    static thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<int> dis(2, N - 1);

    while (true) {
        int x = dis(gen);
        if (gcd(x, N) == 1) {
            return x;
        }
    }
}

bool mathUtils::isMultiplePI(double theta, double eps) {
    double quotient = theta / M_PI;
    double roundedQuotient = round(quotient);
    return abs(quotient - roundedQuotient) < eps;
}

bool mathUtils::isZERO(const complex<double>& z) {
    return z.real() == .0 && z.imag() == .0;
}

