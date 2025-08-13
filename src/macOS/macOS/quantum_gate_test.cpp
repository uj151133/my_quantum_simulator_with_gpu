//
//  quantum_gate_test.cpp
//  macOS
//
//  C++からアダマールゲート専用システムをテスト
//

#include <iostream>
#include <iomanip>
#include "QuantumGateManager.h"

// 実数・虚数分離行列を表示
void print_matrix_split(const Matrix2x2Split& matrix) {
    std::cout << "実数部:\n";
    std::cout << "[\n";
    for (int i = 0; i < 2; i++) {
        std::cout << "  [";
        for (int j = 0; j < 2; j++) {
            std::cout << std::fixed << std::setprecision(6) << matrix.real[i][j];
            if (j < 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "]\n";
    
    std::cout << "虚数部:\n";
    std::cout << "[\n";
    for (int i = 0; i < 2; i++) {
        std::cout << "  [";
        for (int j = 0; j < 2; j++) {
            std::cout << std::fixed << std::setprecision(6) << matrix.imag[i][j];
            if (j < 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "]\n";
}

// メイン関数：アダマールゲートのテスト
int test_hadamard_gate() {
    std::cout << "=== アダマールゲート専用テスト ===\n\n";
    
    // ハッシュテーブルの初期化
    initialize_hadamard_hash_table();
    
    // アダマールゲートのテスト
    std::cout << "アダマールゲート (H):\n";
    uint64_t hadamard_hash = create_hadamard_gate_hash();
    std::cout << "ハッシュ値: 0x" << std::hex << hadamard_hash << std::dec << "\n";
    
    Matrix2x2Split hadamard_matrix;
    if (get_hadamard_matrix_from_hash(hadamard_hash, &hadamard_matrix)) {
        std::cout << "アダマール行列（実数部・虚数部分離）:\n";
        print_matrix_split(hadamard_matrix);
        
        // 理論値との比較
        const float inv_sqrt2 = 1.0f / sqrt(2.0f);
        std::cout << "理論値: 1/√2 = " << std::fixed << std::setprecision(6) << inv_sqrt2 << "\n";
        
        // 精度チェック
        float epsilon = 1e-6f;
        bool correct = 
            abs(hadamard_matrix.real[0][0] - inv_sqrt2) < epsilon &&
            abs(hadamard_matrix.real[0][1] - inv_sqrt2) < epsilon &&
            abs(hadamard_matrix.real[1][0] - inv_sqrt2) < epsilon &&
            abs(hadamard_matrix.real[1][1] - (-inv_sqrt2)) < epsilon;
        
        std::cout << "精度チェック: " << (correct ? "OK" : "NG") << "\n";
    } else {
        std::cout << "エラー: ハッシュ値からゲートを取得できませんでした\n";
    }
    std::cout << "\n";
    
    // 同じゲートを再度作成してハッシュ値が同じことを確認
    std::cout << "ハッシュ値の一意性テスト:\n";
    uint64_t hadamard_hash2 = create_hadamard_gate_hash();
    std::cout << "1回目のアダマールゲートハッシュ: 0x" << std::hex << hadamard_hash << std::dec << "\n";
    std::cout << "2回目のアダマールゲートハッシュ: 0x" << std::hex << hadamard_hash2 << std::dec << "\n";
    std::cout << "同一性チェック: " << (hadamard_hash == hadamard_hash2 ? "OK" : "NG") << "\n";
    std::cout << "\n";
    
    // クリーンアップ
    cleanup_hadamard_hash_table();
    
    std::cout << "=== テスト完了 ===\n";
    return 0;
}

// エントリーポイント
extern "C" int run_hadamard_gate_tests() {
    return test_hadamard_gate();
}
