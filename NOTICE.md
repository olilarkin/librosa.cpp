# NOTICE

librosa.cpp is a C++17 port of [librosa](https://github.com/librosa/librosa),
originally authored by the librosa development team and licensed under the ISC
License (see `LICENSE`). All copyright notices from the upstream project are
preserved. This port inherits and continues that ISC license grant.

- Upstream project: <https://github.com/librosa/librosa>
- Port tracks upstream version: `librosa==0.11.0`
- Port author: Oli Larkin, 2026

See `AUTHORS.md` for the full list of upstream librosa contributors.

## Third-party components (bundled / fetched)

This repository bundles or fetches the following third-party libraries. Their
licenses continue to apply to their respective source files.

| Component | Path | License |
|-----------|------|---------|
| [Eigen](https://gitlab.com/libeigen/eigen) | `modules/eigen/` (git submodule) | MPL2 |
| [fnnls](https://github.com/CERN/TIGRE) — Mikael Twengström, 2021 | `vendor/fnnls/` | MIT (`vendor/fnnls/LICENSE`) |
| incbeta — Lewis Van Winkle | `vendor/incbeta/` | zlib (header of `vendor/incbeta/incbeta.h`) |
| [pffft](https://github.com/marton78/pffft) (marton78 fork) | `modules/pffft/` (git submodule) — only when `LIBROSA_FFT_BACKEND=pffft` | BSD-like (`modules/pffft/LICENSE.txt`) |
| [PocketFFT](https://github.com/mreineck/pocketfft) | `modules/pocketfft/` (git submodule) — only when `LIBROSA_FFT_BACKEND=pocketfft` | BSD-3-Clause (`modules/pocketfft/LICENSE.md`) |
| [GoogleTest](https://github.com/google/googletest) | fetched via CMake `FetchContent` at test-build time | BSD-3-Clause |
| [CLI11](https://github.com/CLIUtils/CLI11) | fetched via CMake `FetchContent` when `LIBROSA_BUILD_CLI=ON` | BSD-3-Clause |

## System libraries linked at build time

These are *not* bundled — they're expected to come from the user's system
(package manager, SDK, etc.). They are linked into the final binary.

| Component | When linked | License | Notes |
|-----------|-------------|---------|-------|
| [libsndfile](http://libsndfile.github.io/libsndfile/) | non-Apple CMake builds when found by `pkg-config` | LGPL-2.1 | Optional audio I/O backend for non-Apple builds. Apple SwiftPM and default Apple CMake builds use AudioToolbox instead. |
| [FFTW3](https://www.fftw.org/) | when `LIBROSA_FFT_BACKEND=fftw` (default on Linux/Windows) | GPL-2.0-or-later | If distributing binaries, consider using the `accelerate` or `pffft` backend, or obtain a non-GPL FFTW commercial license. |
| Apple Accelerate framework | when `LIBROSA_FFT_BACKEND=accelerate` (default on Apple) | Apple SDK terms | System framework, no extra install. |
| Apple AudioToolbox framework | when `LIBROSA_USE_AUDIOTOOLBOX=ON` or when building the Swift package | Apple SDK terms | System audio file I/O framework, no extra install. |

### LGPL notes (libsndfile)

libsndfile is LGPL-2.1. Apple SwiftPM builds do not link it. For non-Apple
CMake builds that do link it, librosa.cpp itself remains ISC, but a binary that
links against an LGPL library inherits LGPL obligations for the combined work.
In practice this means:

- Dynamic linking (the default on all platforms when using system packages)
  satisfies the LGPL naturally — end users can swap the `.so` / `.dylib`.
- Static linking requires you to either provide the librosa.cpp object files
  or written offer to relink against a modified LGPL lib.
- If you redistribute a binary that statically links it, retain its copyright
  notices and make the corresponding source available.

## Citing

If you use librosa.cpp in academic work, please cite the original librosa
project — the algorithms, design decisions, and numerical behaviour being
validated here are all theirs. See the upstream Zenodo record linked from
<https://github.com/librosa/librosa>.
