import Foundation
import Numerics
import Metal
import simd

enum calculation {
    static let device = MTLCreateSystemDefaultDevice()!
    static let library = device.makeDefaultLibrary()!
    static let pipelineState: MTLComputePipelineState = {
        let function = library.makeFunction(name: "hashMatrixElement")!
        return try! device.makeComputePipelineState(function: function)
    }()
    static let commandQueue = device.makeCommandQueue()!

    static func generateUniqueTableKey(node: QMDDNode, row: Int = 0, col: Int = 0, rowStride: Int = 1, colStride: Int = 1, parentWeight: Complex<Double> = Complex<Double>(1.0, 0.0)) ->UInt {
        func customHash(_ c: Complex<Double>) -> UInt {
            let realHash = UInt(bitPattern: c.real.hashValue)
            let imagHash = UInt(bitPattern: c.imaginary.hashValue)
            return realHash ^ (imagHash << 1)
        }

        func hashMatrixElement(value: Complex<Double>, row: Int, col: Int) -> UInt {
            let count = 1
            var values = [SIMD2<Float>(Float(value.real), Float(value.imaginary))]
            var rows = [row]
            var cols = [col]
            var results = [UInt](repeating: 0, count: count)

            let valueBuffer = device.makeBuffer(bytes: &values, length: MemoryLayout<SIMD2<Float>>.stride * count, options: [])
            let rowBuffer = device.makeBuffer(bytes: &rows, length: MemoryLayout<Int>.stride * count, options: [])
            let colBuffer = device.makeBuffer(bytes: &cols, length: MemoryLayout<Int>.stride * count, options: [])
            let resultBuffer = device.makeBuffer(bytes: &results, length: MemoryLayout<UInt>.stride * count, options: [])

            let commandBuffer = commandQueue.makeCommandBuffer()!
            let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
            computeEncoder.setComputePipelineState(pipelineState)
            computeEncoder.setBuffer(valueBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(rowBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(colBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(resultBuffer, offset: 0, index: 3)

            let gridSize = MTLSize(width: count, height: 1, depth: 1)
            let threadGroupSize = MTLSize(width: min(pipelineState.maxTotalThreadsPerThreadgroup, count), height: 1, depth: 1)
            computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)

            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let resultPointer = resultBuffer?.contents().bindMemory(to: UInt.self, capacity: count)
            let resultArray = Array(UnsafeBufferPointer(start: resultPointer, count: count))
            return resultArray[0]
        }

        var hashValue: UInt = 0
        let table = UniqueTable.getInstance()

        for i in 0..<node.edges.count {
            for j in 0..<node.edges[i].count {
                let newRow = row + i * rowStride
                let newCol = col + j * colStride

                let combinedWeight = parentWeight * node.edges[i][j].weight

                let elementHash: UInt
                if node.edges[i][j].isTerminal || node.edges[i][j].uniqueTableKey == 0 {
                    elementHash = hashMatrixElement(value: combinedWeight, row: newRow, col: newCol)
                } else {
                    if let foundNode = table.find(hashKey: node.edges[i][j].uniqueTableKey) {
                        elementHash = calculation.generateUniqueTableKey(node: foundNode, row: newRow, col: newCol, rowStride: rowStride * 2, colStride: colStride * 2, parentWeight: combinedWeight)
                    } else {
                        elementHash = 0
                    }
                }

                hashValue ^= (elementHash + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2))
            }
        }

        return hashValue
    }

    static func generateOperationCacheKey(key: OperationKey) -> Int {
        func customHash(_ c: Complex<Double>) -> Int {
            let realHash = c.real.hashValue
            let imagHash = c.imaginary.hashValue
            return realHash ^ (imagHash << 1)
        }

        func hashCombine(seed: inout Int, hash: Int) {
            seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2)
        }

        var seed = 0
        hashCombine(seed: &seed, hash: customHash(key.0.weight))
        hashCombine(seed: &seed, hash: key.0.uniqueTableKey.hashValue)
        hashCombine(seed: &seed, hash: key.1.hashValue)
        hashCombine(seed: &seed, hash: customHash(key.2.weight))
        hashCombine(seed: &seed, hash: key.2.uniqueTableKey.hashValue)

        return seed
    }
}