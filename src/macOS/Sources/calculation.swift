import Foundation
import Numerics

enum calculation {
    static func generateUniqueTableKey(node: QMDDNode, row: Int = 0, col: Int = 0, rowStride: Int = 1, colStride: Int = 1, parentWeight: Complex<Double> = Complex<Double>(1.0, 0.0)) ->UInt {
        func customHash(_ c: Complex<Double>) -> UInt {
            let realHash = c.real.hashValue
            let imagHash = c.imaginary.hashValue
            return realHash ^ (imagHash << 1)
        }

        func hashMatrixElement(value: Complex<Double>, row: Int, col: Int) -> UInt {
            let valueHash = customHash(value)
            let elementHash = valueHash ^ ((row << 16) | (col & 0xFFFF)) ^ 0x9e3779b9 + (valueHash << 6) + (valueHash >> 2)
            return elementHash
        }

        var hashValu: UInt = 0
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
                    if let foundNode = table.find(node.edges[i][j].uniqueTableKey) {
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