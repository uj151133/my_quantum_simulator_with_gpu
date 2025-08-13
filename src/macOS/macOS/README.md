# アダマールゲート専用ハッシュテーブル管理システム

このプロジェクトは、C++からObjective-C++を介してアダマールゲートの行列を生成し、ハッシュテーブルに登録してそのハッシュ値を返すシステムです。

## 主な特徴

### 1. アダマールゲート専用
- **アダマールゲート (H)**: `H = (1/√2) * [[1, 1], [1, -1]]`
- 実数部・虚数部を分離した高性能な行列表現
- complex.metalを使用しない軽量設計

### 2. 性能最適化
- **実数部・虚数部分離**: `Matrix2x2Split`構造体で実数部・虚数部を別々の配列で管理
- **Metal GPU対応**: GPU並列計算に最適化された設計
- **シンプルなハッシュ関数**: 固定ハッシュ値による高速アクセス

### 3. ハッシュテーブル管理
- Metal上で定義されたハッシュテーブル構造
- C++の`std::unordered_map`を使用したCPU側管理
- 固定ハッシュ値による効率的なアクセス

## ファイル構成

```
macOS/
├── gate.metal               # アダマールゲート行列生成（Metal実装）
├── QuantumGateManager.h     # C++/Objective-C++ヘッダー
├── QuantumGateManager.mm    # Objective-C++実装
├── quantum_gate_test.cpp    # C++テストコード
└── test.swift              # Swiftテストコード
```

## 使用方法

### C++からの呼び出し例

```cpp
#include "QuantumGateManager.h"

// ハッシュテーブルの初期化
initialize_hadamard_hash_table();

// アダマールゲートの作成とハッシュ値取得
uint64_t hadamard_hash = create_hadamard_gate_hash();
printf("アダマールゲートハッシュ: 0x%llx\n", hadamard_hash);

// ハッシュ値から行列を取得
Matrix2x2Split matrix;
if (get_hadamard_matrix_from_hash(hadamard_hash, &matrix)) {
    // 実数部・虚数部分離行列の使用
    printf("H[0,0] = %.6f + %.6fi\n", matrix.real[0][0], matrix.imag[0][0]);
}

// クリーンアップ
cleanup_hadamard_hash_table();
```
```

### Objective-C++からの呼び出し例

```objc
QuantumGateManager* manager = [QuantumGateManager sharedManager];
[manager initializeWithDevice:metalDevice];

// アダマールゲートの作成
uint64_t hash = [manager createHadamardGateHash];

// 行列の取得
Matrix2x2Split matrix;
if ([manager getHadamardMatrixFromHash:hash matrix:&matrix]) {
    // 実数部・虚数部分離行列の使用
}

[manager cleanup];
```

### Swiftからの呼び出し例

```swift
let gateManager = QuantumGateManager.shared()
gateManager.initialize(with: metalDevice)

// アダマールゲートの作成
let hash = gateManager.createHadamardGateHash()

// 行列の取得
var matrix = Matrix2x2Split()
if gateManager.getHadamardMatrixFromHash(hash, matrix: &matrix) {
    // 実数部・虚数部分離行列の使用
}
```

## API リファレンス

### C言語インターフェース

#### 初期化・クリーンアップ
- `void initialize_hadamard_hash_table(void)` - ハッシュテーブルを初期化
- `void cleanup_hadamard_hash_table(void)` - ハッシュテーブルをクリーンアップ

#### アダマールゲート作成
- `uint64_t create_hadamard_gate_hash(void)` - アダマールゲート作成

#### ゲート取得
- `int get_hadamard_matrix_from_hash(uint64_t hash, Matrix2x2Split* matrix)` - ハッシュから行列を取得

### データ構造

#### Matrix2x2Split (実数部・虚数部分離2x2行列)
```c
typedef struct {
    float real[2][2];  // 実数部の行列
    float imag[2][2];  // 虚数部の行列
} Matrix2x2Split;
```

## テストの実行

### C++テストの実行
```cpp
int result = run_hadamard_gate_tests();
```

### Swiftテストの実行
```swift
runSwiftHadamardGateTests()
```

### 統合テストの実行
```swift
runAllHadamardGateTests()
```

## 質問に対する回答

### Q: ハッシュテーブルはMetal上で定義されているか？

**A:** ハッシュテーブルは2つの場所で定義されています：

1. **Metal側** (`gate.metal`): Metal GPU用の固定サイズハッシュテーブル構造体
2. **CPU側** (`QuantumGateManager.mm`): C++の`std::unordered_map`を使用

現在のところ、実際のハッシュテーブル管理はCPU側で行われており、Metal側は構造体定義のみです。

### Q: Metal上でdoubleを実装する方法は？

**A:** Metalでdouble精度を使用する方法：

1. **`#include <metal_float64>`をインクルード**
2. **`simd_double2`、`simd_double3`、`simd_double4`を使用**
3. **Metal Performance Shaders (MPS) を利用**

```metal
#include <metal_stdlib>
#include <metal_float64>  // double精度サポート

using namespace metal;

// double精度複素数
struct ComplexDouble {
    simd_double2 value;  // (real, imag)
};

// double精度2x2行列
struct Matrix2x2Double {
    double real[2][2];
    double imag[2][2];
};
```

外部ライブラリは特に必要ありませんが、double精度はGPUによってはパフォーマンスが大幅に低下する可能性があります。

### Q: 実数部・虚数部分離による性能向上について

**A:** 正しいです！実数部・虚数部を分離することで以下の性能向上が期待できます：

1. **SIMD最適化**: GPU/CPUのベクトル演算を効率的に活用
2. **メモリアクセス最適化**: 連続したメモリアクセスパターン
3. **キャッシュ効率**: データの局所性向上
4. **並列化**: 実数部と虚数部を独立して並列計算可能

## 注意事項

1. **軽量設計**: complex.metalを使用しないため、軽量で高速
2. **アダマールゲート専用**: 他のゲートは削除済み
3. **Metal GPU最適化**: GPU並列計算に特化した設計
4. **ハッシュテーブル**: 現在はCPU側で管理、将来的にMetal側に移行可能
