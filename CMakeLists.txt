cmake_minimum_required(VERSION 3.16)
project(librosa_cpp VERSION 0.1.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Options
option(LIBROSA_BUILD_TESTS "Build tests" ON)
option(LIBROSA_BUILD_EXAMPLES "Build examples" OFF)
option(LIBROSA_BUILD_CROSSVAL_TESTS "Build cross-validation tests against Python librosa" OFF)
option(LIBROSA_BUILD_CLI "Build the librosa CLI tool" OFF)
option(LIBROSA_BUILD_WASM "Build the Emscripten WASM/npm binding" OFF)
option(LIBROSA_BUILD_SWIFT_C_WRAPPER "Build the Swift-friendly C ABI wrapper" OFF)
option(LIBROSA_USE_SOXR "Use libsoxr for explicit SOXR resampling modes" OFF)
if(APPLE)
    option(LIBROSA_USE_AUDIOTOOLBOX "Use Apple AudioToolbox for audio file I/O" ON)
else()
    option(LIBROSA_USE_AUDIOTOOLBOX "Use Apple AudioToolbox for audio file I/O" OFF)
endif()

set(LIBROSA_FFT_BACKEND "auto" CACHE STRING
    "FFT backend: auto, fftw, accelerate, pffft, pocketfft")
set_property(CACHE LIBROSA_FFT_BACKEND PROPERTY
    STRINGS auto fftw accelerate pffft pocketfft)

if(LIBROSA_BUILD_WASM AND NOT EMSCRIPTEN)
    message(FATAL_ERROR "LIBROSA_BUILD_WASM requires configuring with emcmake/Emscripten")
endif()

if(LIBROSA_BUILD_WASM AND LIBROSA_USE_SOXR)
    message(FATAL_ERROR "LIBROSA_USE_SOXR is not supported for Emscripten/WASM builds")
endif()

if(LIBROSA_USE_AUDIOTOOLBOX AND NOT APPLE)
    message(FATAL_ERROR "LIBROSA_USE_AUDIOTOOLBOX is only available on Apple platforms")
endif()

# Eigen3 from submodule
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs in librosa_cpp builds" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable third-party tests in librosa_cpp builds" FORCE)
add_subdirectory(modules/eigen EXCLUDE_FROM_ALL)
unset(BUILD_TESTING CACHE)
unset(BUILD_TESTING)

if(LIBROSA_USE_AUDIOTOOLBOX)
    message(STATUS "librosa audio I/O backend: Apple AudioToolbox")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SNDFILE sndfile)
    if(NOT SNDFILE_FOUND)
        message(STATUS "libsndfile not found; audio I/O (load/write) will throw at runtime")
    endif()
endif()

if(LIBROSA_USE_SOXR)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SOXR REQUIRED soxr)
    message(STATUS "librosa SOXR resampler: enabled (${SOXR_VERSION})")
endif()

# Resolve FFT backend selection.
if(LIBROSA_FFT_BACKEND STREQUAL "auto")
    if(APPLE)
        set(_librosa_fft_backend accelerate)
    else()
        set(_librosa_fft_backend fftw)
    endif()
else()
    set(_librosa_fft_backend ${LIBROSA_FFT_BACKEND})
endif()
message(STATUS "librosa FFT backend: ${_librosa_fft_backend}")

if(_librosa_fft_backend STREQUAL "fftw")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED fftw3)
    set(LIBROSA_FFT_SOURCE src/internal/fft_fftw.cpp)
elseif(_librosa_fft_backend STREQUAL "accelerate")
    set(LIBROSA_FFT_SOURCE src/internal/fft_accelerate.cpp)
elseif(_librosa_fft_backend STREQUAL "pffft")
    set(LIBROSA_FFT_SOURCE src/internal/fft_pffft.cpp)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/modules/pffft/CMakeLists.txt")
        message(FATAL_ERROR
            "LIBROSA_FFT_BACKEND=pffft but modules/pffft is not populated. "
            "Run: git submodule update --init modules/pffft")
    endif()
    # Skip pffft's tests/benchmarks/examples and single-precision code we
    # don't need.
    set(PFFFT_USE_TYPE_FLOAT OFF CACHE BOOL "" FORCE)
    set(PFFFT_USE_TYPE_DOUBLE ON CACHE BOOL "" FORCE)
    set(PFFFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PFFFT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(PFFFT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(INSTALL_PFFFT OFF CACHE BOOL "" FORCE)
    add_subdirectory(modules/pffft EXCLUDE_FROM_ALL)
elseif(_librosa_fft_backend STREQUAL "pocketfft")
    set(LIBROSA_FFT_SOURCE src/internal/fft_pocketfft.cpp)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/modules/pocketfft/pocketfft_hdronly.h")
        message(FATAL_ERROR
            "LIBROSA_FFT_BACKEND=pocketfft but modules/pocketfft is not populated. "
            "Run: git submodule update --init modules/pocketfft")
    endif()
else()
    message(FATAL_ERROR "Unknown LIBROSA_FFT_BACKEND: ${_librosa_fft_backend}")
endif()

# Main library sources
set(LIBROSA_SOURCES
    # Third-party
    vendor/incbeta/incbeta.c

    # FFT backend (selected above)
    ${LIBROSA_FFT_SOURCE}

    # Utilities
    src/util/utils.cpp
    src/util/exceptions.cpp

    # Core modules
    src/core/convert.cpp
    src/core/audio.cpp
    src/core/spectrum.cpp
    src/core/pitch.cpp
    src/core/harmonic.cpp
    src/core/constantq.cpp
    src/core/notation.cpp
    src/core/intervals.cpp

    # Filters
    src/filters.cpp

    # Sequence (Viterbi, DTW)
    src/sequence.cpp

    # Feature extraction
    src/feature/spectral.cpp
    src/feature/rhythm.cpp
    src/feature/utils.cpp
    src/feature/inverse.cpp

    # High-level analysis
    src/onset.cpp
    src/beat.cpp
    src/decompose.cpp
    src/segment.cpp
    src/effects.cpp
)

# Create library
add_library(librosa ${LIBROSA_SOURCES})
add_library(librosa::librosa ALIAS librosa)

target_include_directories(librosa
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(librosa
    PUBLIC
        $<BUILD_INTERFACE:Eigen3::Eigen>
)

if(SNDFILE_FOUND)
    target_compile_definitions(librosa PRIVATE LIBROSA_HAS_SNDFILE)
    target_include_directories(librosa PRIVATE ${SNDFILE_INCLUDE_DIRS})
    target_link_directories(librosa PUBLIC ${SNDFILE_LIBRARY_DIRS})
    target_link_libraries(librosa PUBLIC ${SNDFILE_LIBRARIES})
elseif(LIBROSA_USE_AUDIOTOOLBOX)
    target_compile_definitions(librosa PRIVATE LIBROSA_HAS_AUDIOTOOLBOX)
    target_link_libraries(librosa PUBLIC "-framework AudioToolbox" "-framework CoreFoundation")
endif()

if(LIBROSA_USE_SOXR)
    target_compile_definitions(librosa PUBLIC LIBROSA_HAS_SOXR)
    target_include_directories(librosa PRIVATE ${SOXR_INCLUDE_DIRS})
    target_link_directories(librosa PUBLIC ${SOXR_LIBRARY_DIRS})
    target_link_libraries(librosa PUBLIC ${SOXR_LIBRARIES})
endif()

if(_librosa_fft_backend STREQUAL "fftw")
    target_include_directories(librosa PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_directories(librosa PUBLIC ${FFTW3_LIBRARY_DIRS})
    target_link_libraries(librosa PUBLIC ${FFTW3_LIBRARIES})
elseif(_librosa_fft_backend STREQUAL "accelerate")
    target_link_libraries(librosa PUBLIC "-framework Accelerate")
elseif(_librosa_fft_backend STREQUAL "pffft")
    target_link_libraries(librosa PUBLIC PFFFT::PFFFT)
    if(EMSCRIPTEN)
        target_compile_options(PFFFT PRIVATE -msimd128)
        target_compile_definitions(PFFFT PRIVATE PFFFT_ENABLE_NEON=1)
    endif()
elseif(_librosa_fft_backend STREQUAL "pocketfft")
    target_include_directories(librosa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/modules/pocketfft)
endif()

if(EMSCRIPTEN)
    # Keep SIMD compile flags scoped to PFFFT and the final binding target.
    # Applying -msimd128 to Eigen-heavy librosa.cpp translation units currently
    # trips wasm SelectionDAG crashes in some Emscripten/LLVM releases.
    target_compile_options(librosa PRIVATE -fexceptions)
    target_compile_definitions(librosa PRIVATE EIGEN_DONT_VECTORIZE EIGEN_DONT_ALIGN_STATICALLY)
endif()

if(LIBROSA_BUILD_SWIFT_C_WRAPPER)
    add_library(librosa_c STATIC
        Sources/CLibrosa/librosa_c.cpp
    )
    add_library(librosa::librosa_c ALIAS librosa_c)
    set_target_properties(librosa_c PROPERTIES OUTPUT_NAME CLibrosa)
    target_link_libraries(librosa_c PUBLIC librosa::librosa)
    target_include_directories(librosa_c
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/Sources/CLibrosa/include
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_features(librosa_c PUBLIC cxx_std_17)
endif()

if(LIBROSA_BUILD_WASM)
    if(NOT _librosa_fft_backend STREQUAL "pffft")
        message(FATAL_ERROR "LIBROSA_BUILD_WASM currently requires -DLIBROSA_FFT_BACKEND=pffft")
    endif()

    add_executable(librosa_wasm wasm/bindings.cpp)
    target_link_libraries(librosa_wasm PRIVATE librosa::librosa)
    target_compile_options(librosa_wasm PRIVATE -msimd128 -fexceptions)
    target_compile_definitions(librosa_wasm PRIVATE EIGEN_DONT_VECTORIZE EIGEN_DONT_ALIGN_STATICALLY)
    target_link_options(librosa_wasm PRIVATE
        --bind
        -msimd128
        -fexceptions
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sEXPORT_NAME=createLibrosaModule
        -sENVIRONMENT=web,node
        -sALLOW_MEMORY_GROWTH=1
        -sWASM_BIGINT=1
        -sFILESYSTEM=0
        -sDISABLE_EXCEPTION_CATCHING=0
        -sASSERTIONS=0
    )
    set_target_properties(librosa_wasm PROPERTIES
        OUTPUT_NAME librosa_wasm
        SUFFIX ".mjs"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/packages/librosa-wasm/dist/wasm"
    )
endif()

# Tests
if(LIBROSA_BUILD_TESTS)
    enable_testing()

    # Fetch Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    # Test executable: gate audio-file tests on an available file I/O backend.
    set(_librosa_audio_io_tests "")
    if(SNDFILE_FOUND OR LIBROSA_USE_AUDIOTOOLBOX)
        list(APPEND _librosa_audio_io_tests
            tests/test_audio.cpp
            tests/test_cli.cpp
        )
    endif()

    add_executable(librosa_tests
        tests/test_main.cpp
        tests/test_util.cpp
        tests/test_convert.cpp
        ${_librosa_audio_io_tests}
        tests/test_spectrum.cpp
        tests/test_pitch.cpp
        tests/test_constantq.cpp
        tests/test_filters.cpp
        tests/test_features.cpp
        tests/test_effects.cpp
        tests/test_sequence.cpp
        tests/test_decompose.cpp
        tests/test_onset.cpp
        tests/test_beat.cpp
        tests/test_harmonic.cpp
        tests/test_segment.cpp
        tests/test_notation.cpp
        tests/test_intervals.cpp
        cli/common.cpp
        cli/formatter.cpp
    )

    target_link_libraries(librosa_tests
        PRIVATE
            librosa::librosa
            GTest::gtest
            GTest::gtest_main
    )

    target_include_directories(librosa_tests
        PRIVATE
            ${SNDFILE_INCLUDE_DIRS}
    )
    if(LIBROSA_USE_AUDIOTOOLBOX)
        target_compile_definitions(librosa_tests PRIVATE LIBROSA_HAS_AUDIOTOOLBOX)
    endif()
    if(_librosa_fft_backend STREQUAL "pffft")
        target_compile_definitions(librosa_tests PRIVATE LIBROSA_TEST_FFT_BACKEND_PFFFT)
    endif()

    include(GoogleTest)
    gtest_discover_tests(librosa_tests)

    # Cross-validation tests (optional, requires running generate_references.py first)
    if(LIBROSA_BUILD_CROSSVAL_TESTS)
        add_executable(crossval_tests
            tests/crossval/test_crossval.cpp
        )

        target_link_libraries(crossval_tests
            PRIVATE
                librosa::librosa
                GTest::gtest
                GTest::gtest_main
        )

        # Set working directory for finding reference data
        set_target_properties(crossval_tests PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )

        gtest_discover_tests(crossval_tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endif()

# Examples
if(LIBROSA_BUILD_EXAMPLES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/load_audio.cpp")
        message(FATAL_ERROR
            "LIBROSA_BUILD_EXAMPLES=ON but the examples/ sources are not present in this checkout.")
    endif()

    add_executable(example_load_audio examples/load_audio.cpp)
    target_link_libraries(example_load_audio PRIVATE librosa::librosa)

    add_executable(example_stft examples/stft_example.cpp)
    target_link_libraries(example_stft PRIVATE librosa::librosa)

    add_executable(example_mfcc examples/mfcc_example.cpp)
    target_link_libraries(example_mfcc PRIVATE librosa::librosa)
endif()

# CLI tool
if(LIBROSA_BUILD_CLI)
    include(FetchContent)
    FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.4.2
    )
    FetchContent_MakeAvailable(CLI11)

    add_executable(librosa_cli
        cli/main.cpp
        cli/common.cpp
        cli/formatter.cpp
        cli/commands/info.cpp
        cli/commands/tempo.cpp
        cli/commands/beat.cpp
        cli/commands/onset.cpp
        cli/commands/pitch.cpp
        cli/commands/mfcc.cpp
        cli/commands/melspectrogram.cpp
        cli/commands/chroma.cpp
        cli/commands/spectral.cpp
        cli/commands/rms.cpp
        cli/commands/zcr.cpp
        cli/commands/tonnetz.cpp
        cli/commands/tuning.cpp
        cli/commands/stft.cpp
        cli/commands/trim.cpp
        cli/commands/hpss.cpp
    )

    target_link_libraries(librosa_cli
        PRIVATE
            librosa::librosa
            CLI11::CLI11
    )

    target_include_directories(librosa_cli
        PRIVATE
            ${SNDFILE_INCLUDE_DIRS}
    )
    if(SNDFILE_FOUND)
        target_compile_definitions(librosa_cli PRIVATE LIBROSA_HAS_SNDFILE)
    endif()
    if(LIBROSA_USE_AUDIOTOOLBOX)
        target_compile_definitions(librosa_cli PRIVATE LIBROSA_HAS_AUDIOTOOLBOX)
    endif()

    set_target_properties(librosa_cli PROPERTIES OUTPUT_NAME rosa)

    install(TARGETS librosa_cli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Installation
include(GNUInstallDirs)
# The pffft backend pulls a submodule static-library dependency that can't be
# exposed through our install export set without rehoming pffft's own install
# rules. Skip install for that backend — pffft is used for Emscripten/WASM
# builds where install is unused anyway.
if(NOT _librosa_fft_backend STREQUAL "pffft")
install(TARGETS librosa
    EXPORT librosaTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/librosa
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT librosaTargets
    FILE librosaTargets.cmake
    NAMESPACE librosa::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/librosa
)
endif()
