# librosa.cpp

A C++17 port of the [librosa](https://github.com/librosa/librosa) audio and music
analysis library, built on [Eigen](https://eigen.tuxfamily.org/). Tracks upstream
`librosa==0.11.0`.

> This is a derivative work. All algorithmic credit belongs to the librosa
> development team; this repository re-implements their library in C++. See
> `AUTHORS.md` for the full contributor list and `LICENSE` for the ISC grant
> inherited from upstream.

## Features

- STFT / ISTFT, CQT, mel-spectrograms, chroma
- MFCC and other spectral features (centroid, bandwidth, flatness, rolloff, contrast)
- Onset detection and onset strength
- Tempo estimation and beat tracking
- HPSS (harmonic-percussive source separation), NMF decomposition
- Pitch (YIN, pYIN), tuning estimation
- Viterbi decoding, DTW, sequence utilities
- Effects (time stretching, pitch shifting, trimming, remixing)
- `rosa` CLI for quick analysis from the terminal
- SwiftPM package with a Swift API and C ABI target for Apple platforms
- WebAssembly/npm package with a TypeScript API and browser demo

Correctness is pinned to the Python reference via a cross-validation harness
that generates JSON reference outputs with `librosa==0.11.0` and diffs the C++
implementation against them — see [Cross-validation](#cross-validation) below.

## Dependencies

Build-time:

- CMake ≥ 3.16, a C++17 compiler, `pkg-config`
- Audio file I/O:
  - Apple platforms use AudioToolbox by default; no `libsndfile` install is
    required for SwiftPM or default Apple CMake builds.
  - Non-Apple CMake builds use [libsndfile](http://libsndfile.github.io/libsndfile/)
    when available.
- An FFT backend — see [FFT backend](#fft-backend). Default is FFTW3 on
  Linux/Windows and Apple's Accelerate framework (zero extra dependency) on
  macOS.
- An internal Kaiser-windowed sinc resampler provides the `kaiser_*` modes used
  by the CQT/default resample path, including `kaiser_hq`. No libsoxr
  install or LGPL resampler link is required.

Bundled (no action needed):

- Eigen (git submodule at `modules/eigen`)
- `fnnls`, `incbeta` (vendored in `vendor/`)
- GoogleTest and CLI11 fetched on demand via CMake `FetchContent`

On Ubuntu:

```bash
sudo apt-get install cmake ninja-build pkg-config \
    libsndfile1-dev libfftw3-dev
```

On macOS with Homebrew (FFTW is only needed if you override the default
Accelerate backend):

```bash
brew install cmake ninja
```

## Build

```bash
git clone --recurse-submodules https://github.com/olilarkin/librosa.cpp.git
cd librosa.cpp

cmake -S . -B build -G Ninja \
    -DLIBROSA_BUILD_TESTS=ON \
    -DLIBROSA_BUILD_CLI=ON
cmake --build build
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## FFT backend

librosa.cpp can be built against one of four interchangeable FFT backends,
selected at configure time with `-DLIBROSA_FFT_BACKEND=<backend>`:

| Value         | Library                                         | Notes                                                 |
|---------------|-------------------------------------------------|-------------------------------------------------------|
| `auto` (default) | Accelerate on Apple, otherwise FFTW3         | Zero-config for typical dev machines.                 |
| `fftw`        | [FFTW3](https://www.fftw.org/) (double precision) | Default everywhere except Apple platforms.            |
| `accelerate`  | Apple Accelerate / vDSP                          | macOS/iOS only; no extra dependency.                  |
| `pocketfft`   | [mreineck/pocketfft](https://github.com/mreineck/pocketfft) (double precision) | Header-only C++, no system FFT library. Same FFT NumPy / SciPy use internally. Supports arbitrary transform lengths (including primes > 5) via Bluestein. Good default when avoiding FFTW's GPL or when building for Emscripten/WASM. |
| `pffft`       | [marton78/pffft](https://github.com/marton78/pffft) (double precision) | Pure C with SIMD. Restricted to FFT lengths with small prime factors (2, 3, 5); code paths that require arbitrary-length FFTs (fft-based resample, recursive-downsample CQT) are not available under this backend. Prefer `pocketfft` unless you specifically need pffft's SIMD throughput on 5-smooth sizes. |

Example:

```bash
cmake -S . -B build -G Ninja -DLIBROSA_FFT_BACKEND=pocketfft
```

The `pocketfft` and `pffft` backends require their respective submodule in
`modules/` to be populated (`git submodule update --init --recursive`
handles both).

## Run the test suite

```bash
ctest --test-dir build --output-on-failure
```

The default unit tests are self-contained — they synthesise their own signals
and do not require any external data.

## Swift Package

This repository is also a Swift 5.9+ package for Apple platforms. The package
builds the C++ implementation behind a small C ABI and exposes a Swift API that
uses `[Double]` buffers and a row-major `LibrosaMatrix`.

```swift
dependencies: [
    .package(url: "https://github.com/olilarkin/librosa.cpp.git", branch: "main")
]
```

```swift
.product(name: "Librosa", package: "librosa.cpp")
```

```swift
import Librosa

let audio = try Librosa.load(path: "/path/to/file.wav", sampleRate: 22_050)
let mfcc = try Librosa.mfcc(audio.mono, sampleRate: audio.sampleRate, nMFCC: 13)
print(mfcc.rows, mfcc.columns)
```

SwiftPM Apple builds use Accelerate for FFTs and AudioToolbox for audio file
loading, so they do not link `libsndfile`.

Run the Swift package tests, including Python-reference cross-validation cases:

```bash
swift test
```

To produce a binary C ABI XCFramework for macOS, iOS, iOS Simulator, visionOS,
and visionOS Simulator:

```bash
./scripts/build-xcframework.sh
```

The output is written to `.build/xcframework/CLibrosa.xcframework` and includes
a Clang module header that can be imported from Swift as `CLibrosa`.
Set `LIBROSA_XCFRAMEWORK_PLATFORMS` to a space-separated subset such as
`"macos ios"` when you only want selected slices.

## WebAssembly / npm

The `packages/librosa-wasm` package builds librosa.cpp with Emscripten, the
`pffft` FFT backend, and WASM SIMD enabled:

```bash
cd packages/librosa-wasm
npm install
npm run build
npm test
```

The package entrypoint is asynchronous because it loads a `.wasm` module:

```ts
import { createLibrosa } from "@olilarkin/librosa-wasm";

const librosa = await createLibrosa();
const y = librosa.tone(440, { sr: 22050, duration: 1 });
const mfcc = librosa.mfcc(y, { sr: 22050, nMfcc: 13 });
```

Matrices are row-major `{ rows, cols, data }` objects. Complex matrices use
interleaved real/imaginary `Float64Array` data.

The `examples/web` demo consumes the local package in a browser worker and is
deployed by the `web-pages.yml` workflow. On GitHub releases,
`npm-publish.yml` builds, tests, and publishes `@olilarkin/librosa-wasm` to
GitHub Packages, while `cli-release.yml` attaches packaged native CLI binaries
and a `CLibrosa.xcframework` as release assets.

## Cross-validation

The `tests/crossval/` harness generates reference outputs by running the real
Python librosa (`==0.11.0`) and diffs the C++ implementation against them.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python tests/crossval/generate_references.py      # writes JSON into tests/crossval/reference_data/

cmake -S . -B build -DLIBROSA_BUILD_CROSSVAL_TESTS=ON
cmake --build build
./build/crossval_tests
```

See `tests/crossval/README.md` for the full list of modules covered and how to
add new cases.

## CLI

With `-DLIBROSA_BUILD_CLI=ON` (on by default in CI), a `rosa` binary is
built at `build/rosa`. Usage:

```bash
./build/rosa <file.wav> info
./build/rosa <file.wav> tempo
./build/rosa <file.wav> mfcc --n-mfcc 20
```

Full reference: `cli/CLI.md`.

## Consuming from CMake

```cmake
add_subdirectory(path/to/librosa.cpp)
target_link_libraries(my_target PRIVATE librosa::librosa)
```

Or after `cmake --install build`, use the installed `librosaTargets.cmake`.
Installed consumers also need to satisfy the Eigen dependency themselves,
since Eigen is only linked via `BUILD_INTERFACE`:

```cmake
find_package(Eigen3 REQUIRED)
find_package(librosa REQUIRED)
target_link_libraries(my_target PRIVATE librosa::librosa Eigen3::Eigen)
```

## License

ISC, inherited from upstream librosa — see `LICENSE`. Bundled third-party
components retain their own licenses; see `NOTICE.md` for the full inventory.

## Citation

If you use this library in academic work, please cite the original librosa
project. See the upstream Zenodo record linked from
<https://github.com/librosa/librosa>.
