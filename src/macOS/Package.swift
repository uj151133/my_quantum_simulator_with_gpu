// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "macOS",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .executable(name: "macOS", targets: ["macOS"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics.git", from: "0.0.7")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "macOS",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics")
            ],
            resources: [
                .process(String("Sources/calculation.metal")),
                .process(String("Sources/qmdd.metal")),
                .process(String("Sources/gate.metal")),
            ]
        )
    ]
)