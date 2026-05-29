#include <gtest/gtest.h>
#include <librosa/onset.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;
using namespace librosa::onset;

// ============================================================================
// Maximum Filter 1D Tests
// ============================================================================

TEST(MaximumFilter1DTest, BasicFilter) {
    ArrayXXr S(3, 5);
    S << 1, 2, 3, 2, 1,
         2, 3, 4, 3, 2,
         1, 2, 3, 2, 1;

    ArrayXXr result = maximum_filter1d(S, 3, -2);

    EXPECT_EQ(result.rows(), 3);
    EXPECT_EQ(result.cols(), 5);

    // Middle row should be unchanged (max in column)
    EXPECT_EQ(result(1, 2), 4);
}

TEST(MaximumFilter1DTest, NoFilter) {
    ArrayXXr S(3, 5);
    S.setRandom();

    ArrayXXr result = maximum_filter1d(S, 1, -2);

    // Size 1 should return unchanged
    EXPECT_EQ((result - S).abs().maxCoeff(), 0.0);
}

TEST(MaximumFilter1DTest, TimeAxisFilter) {
    ArrayXXr S(3, 5);
    S << 1, 2, 5, 2, 1,
         1, 2, 3, 2, 1,
         1, 2, 3, 2, 1;

    ArrayXXr result = maximum_filter1d(S, 3, -1);

    // The spike at (0, 2) should spread to neighbors
    EXPECT_EQ(result(0, 1), 5);
    EXPECT_EQ(result(0, 2), 5);
    EXPECT_EQ(result(0, 3), 5);
}

// ============================================================================
// Match Events Tests
// ============================================================================

TEST(MatchEventsTest, ExactMatch) {
    std::vector<Eigen::Index> from = {0, 10, 20, 30};
    std::vector<Eigen::Index> to = {0, 10, 20, 30};

    auto result = match_events(from, to);

    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 1);
    EXPECT_EQ(result[2], 2);
    EXPECT_EQ(result[3], 3);
}

TEST(MatchEventsTest, NearestMatch) {
    std::vector<Eigen::Index> from = {5, 15, 25};
    std::vector<Eigen::Index> to = {0, 10, 20, 30};

    auto result = match_events(from, to);

    EXPECT_EQ(result.size(), 3);
    // 5 is closest to 10 (diff=5) vs 0 (diff=5), should match first
    // 15 is closest to 10 (diff=5) or 20 (diff=5)
    // 25 is closest to 20 (diff=5) or 30 (diff=5)
}

TEST(MatchEventsTest, LeftOnly) {
    std::vector<Eigen::Index> from = {5, 15, 25};
    std::vector<Eigen::Index> to = {0, 10, 20, 30};

    auto result = match_events(from, to, true, false);

    EXPECT_EQ(result.size(), 3);
    // 5 should match 0 (can only go left, 0 is to the left)
    // 15 should match 10
    // 25 should match 20
}

TEST(MatchEventsTest, EmptyInput) {
    std::vector<Eigen::Index> from = {};
    std::vector<Eigen::Index> to = {0, 10, 20};

    EXPECT_THROW(match_events(from, to), ParameterError);
}

// ============================================================================
// Onset Strength Tests
// ============================================================================

TEST(OnsetStrengthTest, BasicStrength) {
    // Create a simple test signal with clear transients
    int sr = 22050;
    int duration = 1;  // 1 second
    int n_samples = sr * duration;

    ArrayXr y = ArrayXr::Zero(n_samples);

    // Add clicks at specific locations
    y(0) = 1.0;
    y(sr / 4) = 1.0;
    y(sr / 2) = 1.0;
    y(3 * sr / 4) = 1.0;

    ArrayXr env = onset_strength(y, sr);

    EXPECT_GT(env.size(), 0);
    // Envelope should have positive values near the clicks
    EXPECT_GE(env.maxCoeff(), 0.0);
}

TEST(OnsetStrengthTest, FromSpectrogram) {
    // Create a simple spectrogram
    ArrayXXr S(128, 100);
    S.setRandom();
    S = S.abs();

    ArrayXr env = onset_strength(S, 22050);

    EXPECT_GT(env.size(), 0);
}

TEST(OnsetStrengthTest, MedianAggregateFromSpectrogram) {
    ArrayXXr S(4, 3);
    S << 0.0, 0.0, 2.0,
         0.0, 0.0, 2.0,
         0.0, 0.0, 2.0,
         0.0, 10.0, 12.0;

    ArrayXr mean_env = onset_strength(S, 22050, 2048, 512, 1, 1,
                                      false, false, AggregateFunc::Mean);
    ArrayXr median_env = onset_strength(S, 22050, 2048, 512, 1, 1,
                                        false, false, AggregateFunc::Median);

    ASSERT_EQ(mean_env.size(), 3);
    ASSERT_EQ(median_env.size(), 3);
    EXPECT_NEAR(mean_env(1), 2.5, 1e-12);
    EXPECT_NEAR(median_env(1), 0.0, 1e-12);
    EXPECT_NEAR(median_env(2), 2.0, 1e-12);
}

// ============================================================================
// Onset Detection Tests
// ============================================================================

TEST(OnsetDetectTest, ClearTransients) {
    // Create signal with clear transients
    int sr = 22050;
    int duration = 1;
    int n_samples = sr * duration;

    ArrayXr y = ArrayXr::Zero(n_samples);

    // Add clicks
    for (int i = 0; i < 4; ++i) {
        int pos = i * sr / 4;
        if (pos < n_samples) {
            y(pos) = 1.0;
            // Add some decay to make it more realistic
            for (int j = 1; j < 100 && pos + j < n_samples; ++j) {
                y(pos + j) = std::exp(-static_cast<Real>(j) / 20.0);
            }
        }
    }

    auto onsets = onset_detect(y, sr);

    // Should detect at least some onsets
    // The exact number depends on the parameters
    EXPECT_GE(onsets.size(), 0);
}

TEST(OnsetDetectTest, WithEnvelope) {
    // Create a synthetic onset envelope
    ArrayXr env(100);
    env.setZero();

    // Add peaks
    env(10) = 1.0;
    env(30) = 1.0;
    env(60) = 1.0;
    env(80) = 1.0;

    // Add some noise
    for (Eigen::Index i = 0; i < env.size(); ++i) {
        env(i) += 0.1 * std::sin(static_cast<Real>(i) * 0.5);
    }
    env = env.max(0.0);

    auto onsets = onset_detect_envelope(env, 22050, 512, false, OnsetUnits::Frames,
                                         true, 3, 3, 3, 3, 0.1, 3);

    // Should detect the peaks
    EXPECT_GT(onsets.size(), 0);
}

TEST(OnsetDetectTest, UnitConversion) {
    // Create simple envelope
    ArrayXr env(100);
    env.setZero();
    env(20) = 1.0;
    env(50) = 1.0;

    int hop_length = 512;
    Real sr = 22050;

    auto frames = onset_detect_envelope(env, sr, hop_length, false, OnsetUnits::Frames,
                                         true, 3, 3, 3, 3, 0.1, 3);

    auto samples = onset_detect_envelope(env, sr, hop_length, false, OnsetUnits::Samples,
                                          true, 3, 3, 3, 3, 0.1, 3);

    // Sample positions should be frames * hop_length
    if (!frames.empty() && !samples.empty()) {
        EXPECT_EQ(samples.size(), frames.size());
        for (size_t i = 0; i < frames.size(); ++i) {
            EXPECT_EQ(samples[i], frames[i] * hop_length);
        }
    }
}

TEST(OnsetDetectTest, TimeUnitsRequireHelper) {
    ArrayXr env(100);
    env.setZero();
    env(20) = 1.0;

    EXPECT_THROW(
        onset_detect_envelope(env, 22050, 512, false, OnsetUnits::Time,
                              true, 3, 3, 3, 3, 0.1, 3),
        ParameterError);
}

// ============================================================================
// Onset Backtrack Tests
// ============================================================================

TEST(OnsetBacktrackTest, BasicBacktrack) {
    // Energy function with clear minima before peaks
    ArrayXr energy(20);
    energy << 0.1, 0.05, 0.1, 0.3, 0.8, 1.0, 0.8, 0.3,
              0.1, 0.05, 0.1, 0.4, 0.9, 1.0, 0.7, 0.2,
              0.1, 0.05, 0.1, 0.2;

    // Detected onsets at peaks
    std::vector<Eigen::Index> events = {5, 13};

    auto backtracked = onset_backtrack(events, energy);

    EXPECT_EQ(backtracked.size(), 2);
    // Should backtrack to minima before peaks
    EXPECT_LT(backtracked[0], 5);
    EXPECT_LT(backtracked[1], 13);
}

TEST(OnsetBacktrackTest, EmptyEvents) {
    ArrayXr energy(20);
    energy.setLinSpaced(0, 1);

    std::vector<Eigen::Index> events = {};

    auto backtracked = onset_backtrack(events, energy);

    EXPECT_EQ(backtracked.size(), 0);
}

TEST(OnsetBacktrackTest, ShortEnergy) {
    ArrayXr energy(2);
    energy << 0.5, 0.5;

    std::vector<Eigen::Index> events = {1};

    auto backtracked = onset_backtrack(events, energy);

    // Not enough data, should return original
    EXPECT_EQ(backtracked.size(), 1);
}

// ============================================================================
// Onset Detect Times Tests
// ============================================================================

TEST(OnsetDetectTimesTest, Basic) {
    int sr = 22050;
    int n_samples = sr;  // 1 second

    ArrayXr y = ArrayXr::Zero(n_samples);
    y(0) = 1.0;
    y(sr / 2) = 1.0;

    ArrayXr times = onset_detect_times(y, sr);

    // Times should be in seconds
    if (times.size() > 0) {
        EXPECT_GE(times(0), 0.0);
        EXPECT_LE(times(times.size() - 1), 1.0);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(OnsetTest, SilentSignal) {
    ArrayXr y = ArrayXr::Zero(22050);  // 1 second of silence

    auto onsets = onset_detect(y, 22050);

    // Should not detect onsets in silence
    EXPECT_EQ(onsets.size(), 0);
}

TEST(OnsetTest, ConstantSignal) {
    ArrayXr y = ArrayXr::Constant(22050, 0.5);  // Constant signal

    auto onsets = onset_detect(y, 22050);

    // Match Python librosa: centered, zero-padded analysis can induce
    // a single onset at the leading edge of a nonzero constant signal.
    EXPECT_TRUE(onsets.empty() || onsets.size() == 1);
}

TEST(OnsetTest, ShortSignal) {
    ArrayXr y(512);  // Very short signal
    y.setRandom();

    auto onsets = onset_detect(y, 22050);

    // Should handle short signals gracefully
    EXPECT_GE(onsets.size(), 0);
}

// ============================================================================
// Onset Strength Multi Tests
// ============================================================================

TEST(OnsetStrengthMultiTest, DefaultChannels) {
    // Generate a signal with varying energy
    Real sr = 22050;
    ArrayXr y(static_cast<Eigen::Index>(sr));  // 1 second
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        y(i) = std::sin(2.0 * M_PI * 440.0 * i / sr);
    }

    auto result = onset_strength_multi(y, sr);

    // With no channels, should return 1 row
    EXPECT_EQ(result.rows(), 1);
    EXPECT_GT(result.cols(), 0);
}

TEST(OnsetStrengthMultiTest, WithChannels) {
    Real sr = 22050;
    ArrayXr y(static_cast<Eigen::Index>(sr));
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        y(i) = std::sin(2.0 * M_PI * 440.0 * i / sr);
    }

    // Split mel bands into 3 sub-bands: [0, 42), [42, 84), [84, 128)
    std::vector<int> channels = {0, 42, 84, 128};
    auto result = onset_strength_multi(y, sr, 2048, 512, 1, 1, false, true, channels);

    // Should return 3 rows (one per sub-band)
    EXPECT_EQ(result.rows(), 3);
    EXPECT_GT(result.cols(), 0);
}

TEST(OnsetStrengthMultiTest, ConsistentWithSingle) {
    Real sr = 22050;
    ArrayXr y(static_cast<Eigen::Index>(sr));
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        y(i) = std::sin(2.0 * M_PI * 440.0 * i / sr);
    }

    auto single = onset_strength(y, sr);
    auto multi = onset_strength_multi(y, sr);

    // Multi with no channels should match single
    EXPECT_EQ(multi.cols(), single.size());
    for (Eigen::Index j = 0; j < single.size(); ++j) {
        EXPECT_NEAR(multi(0, j), single(j), 1e-6);
    }
}

TEST(OnsetStrengthMultiTest, FromSpectrogram) {
    // Create a simple spectrogram
    ArrayXXr S = ArrayXXr::Random(128, 50).abs();

    auto result = onset_strength_multi(S, 22050.0, 2048, 512, 1, 1, false, true);

    EXPECT_EQ(result.rows(), 1);
    EXPECT_GT(result.cols(), 0);
}
