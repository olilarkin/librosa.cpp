// swift-tools-version: 5.9

import PackageDescription

let librosaSources: [String] = [
    "Sources/CLibrosa/librosa_c.cpp",
    "vendor/incbeta/incbeta.c",
    "src/internal/fft_accelerate.cpp",
    "src/util/utils.cpp",
    "src/util/exceptions.cpp",
    "src/core/convert.cpp",
    "src/core/audio.cpp",
    "src/core/spectrum.cpp",
    "src/core/pitch.cpp",
    "src/core/harmonic.cpp",
    "src/core/constantq.cpp",
    "src/core/notation.cpp",
    "src/core/intervals.cpp",
    "src/filters.cpp",
    "src/sequence.cpp",
    "src/feature/spectral.cpp",
    "src/feature/rhythm.cpp",
    "src/feature/utils.cpp",
    "src/feature/inverse.cpp",
    "src/onset.cpp",
    "src/beat.cpp",
    "src/decompose.cpp",
    "src/segment.cpp",
    "src/effects.cpp",
]

let package = Package(
    name: "Librosa",
    platforms: [
        .iOS(.v13),
        .macOS(.v12),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "Librosa", targets: ["Librosa"]),
        .library(name: "CLibrosa", targets: ["CLibrosa"]),
    ],
    targets: [
        .target(
            name: "CLibrosa",
            path: ".",
            sources: librosaSources,
            publicHeadersPath: "Sources/CLibrosa/include",
            cSettings: [
                .headerSearchPath("Sources/CLibrosa/include"),
                .headerSearchPath("include"),
                .headerSearchPath("modules/eigen"),
                .headerSearchPath("vendor"),
                .headerSearchPath("src"),
            ],
            cxxSettings: [
                .define("LIBROSA_HAS_AUDIOTOOLBOX"),
                .headerSearchPath("Sources/CLibrosa/include"),
                .headerSearchPath("include"),
                .headerSearchPath("modules/eigen"),
                .headerSearchPath("vendor"),
                .headerSearchPath("src"),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("CoreFoundation"),
            ]
        ),
        .target(
            name: "Librosa",
            dependencies: ["CLibrosa"],
            path: "Sources/Librosa"
        ),
        .testTarget(
            name: "LibrosaTests",
            dependencies: ["Librosa"],
            path: "tests/LibrosaTests",
            resources: [
                .copy("ReferenceData"),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
