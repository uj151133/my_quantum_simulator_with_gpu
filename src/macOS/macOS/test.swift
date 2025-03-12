import Metal
import Foundation

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
