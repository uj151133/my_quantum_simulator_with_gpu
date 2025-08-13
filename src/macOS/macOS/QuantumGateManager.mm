//
//  QuantumGateManager.mm
//  macOS
//
//  アダマールゲート専用の管理システム実装
//

#import "QuantumGateManager.h"
#import <Metal/Metal.h>
#import <unordered_map>
#import <memory>

// C++で実装されたハッシュテーブル（アダマールゲート専用）
static std::unordered_map<uint64_t, Matrix2x2Split>* hadamard_hash_table = nullptr;
static QuantumGateManager* sharedInstance = nil;

// Metal関連の変数
static id<MTLDevice> metalDevice = nil;
static id<MTLLibrary> metalLibrary = nil;

#ifdef __cplusplus
extern "C" {
#endif

// アダマールゲート専用の固定ハッシュ値
const uint64_t HADAMARD_HASH = 0x1234567890ABCDEFULL + 1;

// アダマールゲートの行列を生成（実数部・虚数部分離）
Matrix2x2Split create_hadamard_matrix_split() {
    Matrix2x2Split matrix;
    const float inv_sqrt2 = 1.0f / sqrt(2.0f);  // 1/√2
    
    // アダマール行列: H = (1/√2) * [[1, 1], [1, -1]]
    // 実数部
    matrix.real[0][0] = inv_sqrt2;   // H[0,0] = 1/√2
    matrix.real[0][1] = inv_sqrt2;   // H[0,1] = 1/√2  
    matrix.real[1][0] = inv_sqrt2;   // H[1,0] = 1/√2
    matrix.real[1][1] = -inv_sqrt2;  // H[1,1] = -1/√2
    
    // 虚数部（アダマールゲートは実数のみなので全て0）
    matrix.imag[0][0] = 0.0f;
    matrix.imag[0][1] = 0.0f;
    matrix.imag[1][0] = 0.0f;
    matrix.imag[1][1] = 0.0f;
    
    return matrix;
}

// ハッシュテーブルを初期化
void initialize_hadamard_hash_table(void) {
    if (hadamard_hash_table == nullptr) {
        hadamard_hash_table = new std::unordered_map<uint64_t, Matrix2x2Split>();
    }
}

// ハッシュテーブルをクリーンアップ
void cleanup_hadamard_hash_table(void) {
    if (hadamard_hash_table != nullptr) {
        delete hadamard_hash_table;
        hadamard_hash_table = nullptr;
    }
}

// アダマールゲートを作成してハッシュテーブルに登録
uint64_t create_hadamard_gate_hash(void) {
    if (hadamard_hash_table == nullptr) {
        initialize_hadamard_hash_table();
    }
    
    // 既に存在する場合はそのハッシュを返す
    if (hadamard_hash_table->find(HADAMARD_HASH) != hadamard_hash_table->end()) {
        return HADAMARD_HASH;
    }
    
    Matrix2x2Split matrix = create_hadamard_matrix_split();
    
    // ハッシュテーブルに登録
    (*hadamard_hash_table)[HADAMARD_HASH] = matrix;
    
    return HADAMARD_HASH;
}

// ハッシュ値からアダマールゲートの行列を取得
int get_hadamard_matrix_from_hash(uint64_t hash, Matrix2x2Split* matrix) {
    if (hadamard_hash_table == nullptr) {
        return 0; // エラー：ハッシュテーブルが初期化されていない
    }
    
    auto it = hadamard_hash_table->find(hash);
    if (it == hadamard_hash_table->end()) {
        return 0; // エラー：ハッシュが見つからない
    }
    
    if (matrix != nullptr) {
        *matrix = it->second;
    }
    
    return 1; // 成功
}

#ifdef __cplusplus
}
#endif

// Objective-C++実装
@implementation QuantumGateManager

+ (instancetype)sharedManager {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[self alloc] init];
    });
    return sharedInstance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        initialize_hadamard_hash_table();
    }
    return self;
}

- (void)initializeWithDevice:(id<MTLDevice>)device {
    metalDevice = device;
    
    NSError* error = nil;
    metalLibrary = [device newDefaultLibraryWithBundle:[NSBundle mainBundle] error:&error];
    if (error) {
        NSLog(@"Error creating Metal library: %@", error);
    }
}

- (uint64_t)createHadamardGateHash {
    return create_hadamard_gate_hash();
}

- (BOOL)getHadamardMatrixFromHash:(uint64_t)hash matrix:(Matrix2x2Split*)matrix {
    return get_hadamard_matrix_from_hash(hash, matrix) == 1;
}

- (void)cleanup {
    cleanup_hadamard_hash_table();
    metalDevice = nil;
    metalLibrary = nil;
}

- (void)dealloc {
    [self cleanup];
}

@end
