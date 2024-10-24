// swift-tools-version:5.3
import PackageDescription

let package = Package(
    name: "YourProjectName",
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics.git", from: "0.0.7"),
    ],
    targets: [
        .target(
            name: "YourProjectName",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics"),
            ]),
    ]
)