//
//  QuantumGateManager.h
//  macOS
//
//  アダマールゲート専用の管理システム
//

#ifndef QuantumGateManager_h
#define QuantumGateManager_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifdef __cplusplus
extern "C" {
#endif

// 実数部・虚数部を分離した2x2行列
typedef struct {
    float real[2][2];  // 実数部の行列
    float imag[2][2];  // 虚数部の行列
} Matrix2x2Split;

// アダマールゲート専用の関数
uint64_t create_hadamard_gate_hash(void);
int get_hadamard_matrix_from_hash(uint64_t hash, Matrix2x2Split* matrix);
void initialize_hadamard_hash_table(void);
void cleanup_hadamard_hash_table(void);

#ifdef __cplusplus
}
#endif

// Objective-C++インターフェース（アダマールゲート専用）
@interface QuantumGateManager : NSObject

+ (instancetype)sharedManager;
- (void)initializeWithDevice:(id<MTLDevice>)device;
- (uint64_t)createHadamardGateHash;
- (BOOL)getHadamardMatrixFromHash:(uint64_t)hash matrix:(Matrix2x2Split*)matrix;
- (void)cleanup;

@end

#endif /* QuantumGateManager_h */
