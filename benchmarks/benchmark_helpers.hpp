#pragma once

#include <benchmark/benchmark.h>

#include <librosa/core/audio.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/types.hpp>

#include <Eigen/Sparse>

#include <cmath>
#include <complex>
#include <optional>
#include <utility>
#include <vector>

namespace librosa_bench {

constexpr librosa::Real kSampleRate = 22050.0;
constexpr int kHopLength = 512;
constexpr int kNFFT = 2048;

inline librosa::ArrayXr make_tone(librosa::Real frequency,
                                  Eigen::Index n_samples,
                                  librosa::Real sr = kSampleRate) {
    librosa::ArrayXr y(n_samples);
    for (Eigen::Index i = 0; i < n_samples; ++i) {
        y(i) = std::sin(2.0 * librosa::constants::PI * frequency *
                        static_cast<librosa::Real>(i) / sr);
    }
    return y;
}

inline librosa::ArrayXr make_chirp(Eigen::Index n_samples,
                                   librosa::Real sr = kSampleRate) {
    return librosa::chirp(220.0, 1760.0, sr, static_cast<int>(n_samples),
                          std::nullopt, true);
}

inline librosa::ArrayXr make_click_track(Eigen::Index n_samples,
                                         int period_frames = 22,
                                         int hop_length = kHopLength) {
    librosa::ArrayXr y = librosa::ArrayXr::Zero(n_samples);
    const Eigen::Index n_frames = n_samples / hop_length;

    for (Eigen::Index frame = 0; frame < n_frames; frame += period_frames) {
        const Eigen::Index sample = frame * hop_length;
        if (sample < n_samples) {
            y(sample) = 1.0;
            for (Eigen::Index i = 1; i < 64 && sample + i < n_samples; ++i) {
                y(sample + i) = std::exp(-static_cast<librosa::Real>(i) / 10.0);
            }
        }
    }

    return y;
}

inline librosa::ArrayXr make_onset_envelope(Eigen::Index n_frames,
                                            int period = 22) {
    librosa::ArrayXr envelope(n_frames);

    for (Eigen::Index i = 0; i < n_frames; ++i) {
        const bool beat = (i % period) == 0;
        const librosa::Real slow_mod =
            0.1 * std::sin(2.0 * librosa::constants::PI *
                           static_cast<librosa::Real>(i) / 97.0);
        envelope(i) = beat ? 1.0 + slow_mod : 0.03;
    }

    return envelope;
}

inline librosa::ArrayXr make_random_vector(Eigen::Index n) {
    librosa::ArrayXr x(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const librosa::Real a = std::sin(static_cast<librosa::Real>(i) * 0.17);
        const librosa::Real b = std::cos(static_cast<librosa::Real>(i) * 0.031);
        x(i) = 0.6 * a + 0.4 * b;
    }
    return x;
}

inline librosa::ArrayXXr make_feature_matrix(Eigen::Index rows,
                                             Eigen::Index cols) {
    librosa::ArrayXXr x(rows, cols);
    for (Eigen::Index r = 0; r < rows; ++r) {
        for (Eigen::Index c = 0; c < cols; ++c) {
            x(r, c) = std::sin(0.03 * static_cast<librosa::Real>((r + 1) * (c + 1))) +
                      0.5 * std::cos(0.11 * static_cast<librosa::Real>(r + c));
        }
    }
    return x;
}

inline librosa::ArrayXXr make_positive_matrix(Eigen::Index rows,
                                              Eigen::Index cols) {
    return make_feature_matrix(rows, cols).abs() + 0.01;
}

inline librosa::ArrayXXc make_complex_matrix(Eigen::Index rows,
                                             Eigen::Index cols) {
    librosa::ArrayXXc x(rows, cols);
    for (Eigen::Index r = 0; r < rows; ++r) {
        for (Eigen::Index c = 0; c < cols; ++c) {
            const librosa::Real mag =
                0.1 + std::abs(std::sin(0.07 * static_cast<librosa::Real>((r + 1) * (c + 1))));
            const librosa::Real phase = 0.013 * static_cast<librosa::Real>(r * c);
            x(r, c) = std::polar(mag, phase);
        }
    }
    return x;
}

inline librosa::SparseMatrixXr make_neighbor_graph(Eigen::Index n) {
    std::vector<Eigen::Triplet<librosa::Real>> triplets;
    triplets.reserve(static_cast<size_t>(n) * 2);
    for (Eigen::Index i = 0; i < n; ++i) {
        if (i > 0) {
            triplets.emplace_back(i, i - 1, 1.0);
        }
        if (i + 1 < n) {
            triplets.emplace_back(i, i + 1, 1.0);
        }
    }

    librosa::SparseMatrixXr rec(n, n);
    rec.setFromTriplets(triplets.begin(), triplets.end());
    return rec;
}

inline librosa::ArrayXXr make_recurrence(Eigen::Index n) {
    librosa::ArrayXXr rec = librosa::ArrayXXr::Zero(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        rec(i, i) = 1.0;
        if (i >= 4) {
            rec(i, i - 4) = 1.0;
            rec(i - 4, i) = 1.0;
        }
    }
    return rec;
}

template <typename Derived>
inline void consume_eigen(const Eigen::DenseBase<Derived>& value) {
    const auto& derived = value.derived();
    benchmark::DoNotOptimize(derived.data());
    benchmark::DoNotOptimize(derived.rows());
    benchmark::DoNotOptimize(derived.cols());
    benchmark::ClobberMemory();
}

template <typename Scalar, int Options, typename StorageIndex>
inline void consume_sparse(
    const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& value) {
    benchmark::DoNotOptimize(value.valuePtr());
    benchmark::DoNotOptimize(value.rows());
    benchmark::DoNotOptimize(value.cols());
    benchmark::DoNotOptimize(value.nonZeros());
    benchmark::ClobberMemory();
}

template <typename T>
inline void consume_vector(const std::vector<T>& value) {
    benchmark::DoNotOptimize(value.data());
    benchmark::DoNotOptimize(value.size());
    benchmark::ClobberMemory();
}

template <typename A, typename B>
inline void consume_pair(const std::pair<A, B>& value) {
    benchmark::DoNotOptimize(value.first);
    benchmark::DoNotOptimize(value.second);
    benchmark::ClobberMemory();
}

} // namespace librosa_bench
