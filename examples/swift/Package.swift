// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "LibrosaSwiftExample",
    platforms: [
        .macOS(.v12),
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "LibrosaSwiftExample",
            dependencies: [
                .product(name: "Librosa", package: "librosa.cpp"),
            ]
        ),
    ]
)
