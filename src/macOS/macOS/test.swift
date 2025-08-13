//
//  test.swift
//  macOS
//
//  Swiftからアダマールゲート専用システムをテスト
//

import Metal
import Foundation

class HadamardGateTest {
    private let gateManager = QuantumGateManager.shared()
    
    init() {
        // Metalデバイスの初期化
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metalデバイスを作成できませんでした")
        }
        gateManager.initialize(with: device)
    }
    
    func runTests() {
        print("=== Swift側からアダマールゲートテスト開始 ===")
        
        testHadamardGate()
        testHashUniqueness()
        
        print("=== Swift側テスト完了 ===")
    }
    
    private func testHadamardGate() {
        print("\nアダマールゲートのテスト:")
        
        let hash = gateManager.createHadamardGateHash()
        print("ハッシュ値: 0x\(String(hash, radix: 16))")
        
        var matrix = Matrix2x2Split()
        
        if gateManager.getHadamardMatrixFromHash(hash, matrix: &matrix) {
            print("アダマール行列（実数部・虚数部分離）:")
            printMatrixSplit(matrix)
            
            // 理論値との比較
            let invSqrt2 = Float(1.0 / sqrt(2.0))
            print("理論値: 1/√2 = \(String(format: "%.6f", invSqrt2))")
            
            // 精度チェック
            let epsilon: Float = 1e-6
            let correct = 
                abs(matrix.real.0.0 - invSqrt2) < epsilon &&
                abs(matrix.real.0.1 - invSqrt2) < epsilon &&
                abs(matrix.real.1.0 - invSqrt2) < epsilon &&
                abs(matrix.real.1.1 - (-invSqrt2)) < epsilon
            
            print("精度チェック: \(correct ? "OK" : "NG")")
        } else {
            print("エラー: ハッシュからゲートを取得できませんでした")
        }
    }
    
    private func testHashUniqueness() {
        print("\nハッシュ値の一意性テスト:")
        
        let hash1 = gateManager.createHadamardGateHash()
        let hash2 = gateManager.createHadamardGateHash()
        
        print("1回目のアダマールゲートハッシュ: 0x\(String(hash1, radix: 16))")
        print("2回目のアダマールゲートハッシュ: 0x\(String(hash2, radix: 16))")
        print("同一性チェック: \(hash1 == hash2 ? "OK" : "NG")")
    }
    
    private func printMatrixSplit(_ matrix: Matrix2x2Split) {
        print("実数部:")
        print("[")
        print("  [\(String(format: "%.6f", matrix.real.0.0)), \(String(format: "%.6f", matrix.real.0.1))]")
        print("  [\(String(format: "%.6f", matrix.real.1.0)), \(String(format: "%.6f", matrix.real.1.1))]")
        print("]")
        
        print("虚数部:")
        print("[")
        print("  [\(String(format: "%.6f", matrix.imag.0.0)), \(String(format: "%.6f", matrix.imag.0.1))]")
        print("  [\(String(format: "%.6f", matrix.imag.1.0)), \(String(format: "%.6f", matrix.imag.1.1))]")
        print("]")
    }
    
    deinit {
        gateManager.cleanup()
    }
}

// 外部から呼び出し可能な関数
@_cdecl("runSwiftHadamardGateTests")
public func runSwiftHadamardGateTests() {
    let test = HadamardGateTest()
    test.runTests()
}

// C++テスト関数の宣言
@_silgen_name("run_hadamard_gate_tests")
func runCppHadamardGateTests() -> Int32

// 統合テスト関数
func runAllHadamardGateTests() {
    print("=== 統合アダマールゲートテスト開始 ===")
    
    // C++側のテスト実行
    print("\n--- C++側テスト ---")
    let cppResult = runCppHadamardGateTests()
    print("C++テスト結果: \(cppResult == 0 ? "成功" : "失敗")")
    
    // Swift側のテスト実行
    print("\n--- Swift側テスト ---")
    runSwiftHadamardGateTests()
    
    print("\n=== 統合アダマールゲートテスト完了 ===")
}

// 元のMetal GPU テスト用のコードも残しておく
func runOriginalMetalTest() {
    // Metal デバイスを作成
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!

    // シェーダライブラリを読み込む
    let metallibPath = URL(fileURLWithPath: "/Users/mitsuishikaito/my_quantum_simulator_with_gpu/src/macOS/macOS/gate.metallib")
    do {
        let library = try device.makeLibrary(URL: metallibPath)
        
        // シェーダ関数を取得
        guard let function = library.makeFunction(name: "ZEROMatrix") else {
            fatalError("Failed to get function from library")
        }
        
        // パイプラインステートを作成
        let pipelineState: MTLComputePipelineState
        do {
            pipelineState = try device.makeComputePipelineState(function: function)
        } catch {
            fatalError("Failed to create compute pipeline state: \(error)")
        }

        // 入力データの準備
        struct Complex {
            var real: Float
            var imag: Float
        }

        struct ComplexMatrix {
            var rows: UInt32
            var cols: UInt32
            var data: [Complex]
        }

        var matrix = ComplexMatrix(
            rows: 2,
            cols: 2,
            data: [
                Complex(real: 1.0, imag: 2.0),
                Complex(real: 3.0, imag: 4.0),
                Complex(real: 5.0, imag: 6.0),
                Complex(real: 7.0, imag: 8.0)
            ]
        )

        // Metal バッファを作成
        let matrixSize = MemoryLayout<ComplexMatrix>.stride + MemoryLayout<Complex>.stride * matrix.data.count
        let matrixBuffer = device.makeBuffer(bytes: &matrix, length: matrixSize, options: .storageModeShared)!

        // コマンドバッファを作成
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        // パイプラインステートを設定
        computeEncoder.setComputePipelineState(pipelineState)
        
        // バッファを設定
        computeEncoder.setBuffer(matrixBuffer, offset: 0, index: 0)

        // グリッドとスレッドグループの設定
        let gridSize = MTLSize(width: Int(matrix.cols), height: Int(matrix.rows), depth: 1)
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

        // 計算をディスパッチ
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

        // コマンドのエンコードが完了したら
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // 計算結果を表示
        let result = matrixBuffer.contents().assumingMemoryBound(to: ComplexMatrix.self).pointee
        print("Updated Matrix:")
        for i in 0..<Int(result.rows) {
            for j in 0..<Int(result.cols) {
                let index = i * Int(result.cols) + j
                let complexValue = result.data[index]
                print("(\(complexValue.real), \(complexValue.imag))", terminator: " ")
            }
            print()
        }

    } catch {
        fatalError("Failed to create Metal library: \(error)")
    }
}
