#include <gtest/gtest.h>
#include <librosa/core/harmonic.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;
using namespace librosa::core;

// ============================================================================
// interp_harmonics tests
// ============================================================================

TEST(InterpHarmonicsTest, Basic2D) {
    // Create a simple spectrogram with a peak at bin 10
    Eigen::Index n_freq = 20;
    Eigen::Index n_frames = 3;
    ArrayXXr S = ArrayXXr::Zero(n_freq, n_frames);
    S(10, 0) = 1.0;
    S(10, 1) = 1.0;
    S(10, 2) = 1.0;

    // Linear frequency axis
    ArrayXr freqs(n_freq);
    for (Eigen::Index i = 0; i < n_freq; ++i) {
        freqs(i) = i * 100.0;  // 0, 100, 200, ..., 1900
    }

    std::vector<Real> harmonics = {1.0, 2.0};

    ArrayXXr result = interp_harmonics(S, freqs, harmonics);

    // Should have shape (2*20, 3)
    EXPECT_EQ(result.rows(), 2 * n_freq);
    EXPECT_EQ(result.cols(), n_frames);

    // First harmonic (h=1): should match original
    EXPECT_NEAR(result(10, 0), 1.0, 1e-10);

    // Second harmonic (h=2): peak at bin 10 maps to freq 2000,
    // which is at bin index 20 (out of range), so should be fill_value=0
    // But for the fundamental peak at freq 1000, the 2x harmonic lands at 2000
    // which is beyond our frequency range [0, 1900]
}

TEST(InterpHarmonicsTest, Basic1D) {
    Eigen::Index n_freq = 10;
    ArrayXr x(n_freq);
    ArrayXr freqs(n_freq);

    for (Eigen::Index i = 0; i < n_freq; ++i) {
        freqs(i) = i * 100.0;
        x(i) = i * 1.0;
    }

    std::vector<Real> harmonics = {1.0, 0.5};

    ArrayXXr result = interp_harmonics(x, freqs, harmonics);

    // Shape: (2, 10)
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), n_freq);

    // h=1: identity mapping
    for (Eigen::Index i = 0; i < n_freq; ++i) {
        EXPECT_NEAR(result(0, i), x(i), 1e-10);
    }
}

TEST(InterpHarmonicsTest, EmptyHarmonics) {
    ArrayXXr S = ArrayXXr::Ones(5, 3);
    ArrayXr freqs = ArrayXr::LinSpaced(5, 0, 400);

    EXPECT_THROW(interp_harmonics(S, freqs, {}), ParameterError);
}

TEST(InterpHarmonicsTest, MismatchedSize) {
    ArrayXXr S = ArrayXXr::Ones(5, 3);
    ArrayXr freqs = ArrayXr::LinSpaced(10, 0, 900);  // Wrong size

    EXPECT_THROW(interp_harmonics(S, freqs, {1.0}), ParameterError);
}

// ============================================================================
// f0_harmonics tests
// ============================================================================

TEST(F0HarmonicsTest, ConstantF0) {
    Eigen::Index n_freq = 50;
    Eigen::Index n_frames = 5;

    // Create spectrogram with peak at bin corresponding to 440 Hz
    ArrayXr freqs = ArrayXr::LinSpaced(n_freq, 0, 4900);
    ArrayXXr S = ArrayXXr::Zero(n_freq, n_frames);

    // Put energy at the bin closest to 440 Hz
    for (Eigen::Index t = 0; t < n_frames; ++t) {
        S(4, t) = 1.0;  // ~400 Hz
        S(9, t) = 0.5;  // ~900 Hz (approximate 2nd harmonic)
    }

    // Constant f0 at 440 Hz
    ArrayXr f0 = ArrayXr::Constant(n_frames, 440.0);

    std::vector<Real> harmonics = {1.0, 2.0};

    ArrayXXr result = f0_harmonics(S, f0, freqs, harmonics);

    // Shape: (2, 5)
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), n_frames);
}

TEST(F0HarmonicsTest, MismatchedFrames) {
    ArrayXXr S = ArrayXXr::Ones(10, 5);
    ArrayXr f0 = ArrayXr::Ones(3);  // Wrong size
    ArrayXr freqs = ArrayXr::LinSpaced(10, 0, 900);

    EXPECT_THROW(f0_harmonics(S, f0, freqs, {1.0}), ParameterError);
}

// ============================================================================
// salience tests
// ============================================================================

TEST(SalienceTest, BasicSalience) {
    Eigen::Index n_freq = 20;
    Eigen::Index n_frames = 3;

    ArrayXr freqs = ArrayXr::LinSpaced(n_freq, 0, 1900);
    ArrayXXr S = ArrayXXr::Zero(n_freq, n_frames);

    // Put a peak at frequency bin 5 (500 Hz)
    S(5, 0) = 1.0;
    S(5, 1) = 1.0;
    S(5, 2) = 1.0;

    std::vector<Real> harmonics = {1.0, 2.0};

    ArrayXXr sal = salience(S, freqs, harmonics);

    // Shape should match input
    EXPECT_EQ(sal.rows(), n_freq);
    EXPECT_EQ(sal.cols(), n_frames);
}

TEST(SalienceTest, WithWeights) {
    Eigen::Index n_freq = 10;
    ArrayXr freqs = ArrayXr::LinSpaced(n_freq, 0, 900);
    ArrayXXr S = ArrayXXr::Ones(n_freq, 2);

    std::vector<Real> harmonics = {1.0, 2.0};
    std::vector<Real> weights = {0.8, 0.2};

    EXPECT_NO_THROW(salience(S, freqs, harmonics, weights));
}

TEST(SalienceTest, NoFilterPeaks) {
    Eigen::Index n_freq = 10;
    ArrayXr freqs = ArrayXr::LinSpaced(n_freq, 0, 900);
    ArrayXXr S = ArrayXXr::Ones(n_freq, 2);

    std::vector<Real> harmonics = {1.0};

    ArrayXXr sal = salience(S, freqs, harmonics, {}, false);

    // Without peak filtering, all values should be non-zero
    // (since S is all ones and h=1 maps identity)
    EXPECT_GT(sal.sum(), 0);
}
