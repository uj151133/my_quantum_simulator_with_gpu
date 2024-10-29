import Foundation

class OperationCache {
    private var cache: [UInt: OperationResult] = [:]
    private let cacheQueue = DispatchQueue(label: "operationCache.queue", attributes: .concurrent)

    private init() {}

    static let shared = OperationCache()

    func insert(cacheKey: UInt, result: OperationResult) {
        cacheQueue.async(flags: .barrier) {
            self.cache[cacheKey] = result
        }
    }

    func find(cacheKey: UInt) -> OperationResult? {
        var result: OperationResult?
        cacheQueue.sync {
            result = self.cache[cacheKey]
        }
        return result
    }

    func printAllEntries() {
        cacheQueue.sync {
            for (key, result) in self.cache {
                print("Key: \(key)")
                print("Result: \(result.0) \(result.1)")
            }
        }
    }
}