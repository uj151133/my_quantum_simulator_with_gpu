import Foundation
import Numerics

class UniqueTable {
    static let instance = UniqueTable()
    private var table: [UInt: [QMDDNode]] = [:]
    private let tableQueue = DispatchQueue(label: "uniqueTable.queue", attributes: .concurrent)

    private init() {}

    static func getInstance() -> UniqueTable {
        return instance
    }

    func insert(hashKey: UInt, node: QMDDNode) {
        tableQueue.async(flags: .barrier) {
            if self.table[hashKey] != nil {
                self.table[hashKey]?.append(node)
            } else {
                self.table[hashKey] = [node]
            }
        }
    }

    func find(hashKey: UInt) -> QMDDNode? {
        var result: QMDDNode?
        tableQueue.sync {
            if let nodes = self.table[hashKey], !nodes.isEmpty {
                result = nodes[0]
            }
        }
        return result
    }

    func printAllEntries() {
        tableQueue.sync {
            for (key, nodes) in self.table {
                print("Key: \(key)")
                print("Nodes:")

                for node in nodes {
                    print("  \(node)")
                }
                print()
            }
        }
    }
}