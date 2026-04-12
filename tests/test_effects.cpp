#include <gtest/gtest.h>
#include <librosa/effects.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/feature/spectral.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;
using namespace librosa::effects;

// Helper to generate test signals
namespace {

ArrayXr generate_sine(Real freq, Real sr, Real duration) {
    int n_samples = static_cast<int>(sr * duration);
    ArrayXr y(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        y(i) = std::sin(2 * M_PI * freq * i / sr);
    }
    return y;
}

ArrayXr generate_silence(int n_samples) {
    return ArrayXr::Zero(n_samples);
}

ArrayXr generate_signal_with_silence(Real sr, Real duration) {
    // Create signal with silence at start and end
    int n_samples = static_cast<int>(sr * duration);
    int silence_samples = n_samples / 4;
    int signal_samples = n_samples - 2 * silence_samples;

    ArrayXr y(n_samples);
    y.head(silence_samples).setZero();
    for (int i = 0; i < signal_samples; ++i) {
        y(silence_samples + i) = std::sin(2 * M_PI * 440.0 * i / sr);
    }
    y.tail(silence_samples).setZero();

    return y;
}

} // anonymous namespace

// ============================================================================
// Time Stretch Tests
// ============================================================================

TEST(TimeStretchTest, BasicStretch) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    // Speed up by 2x
    ArrayXr y_fast = time_stretch(y, 2.0);

    // Should be approximately half the length
    EXPECT_NEAR(static_cast<double>(y_fast.size()), y.size() / 2.0, y.size() * 0.1);
}

TEST(TimeStretchTest, SlowDown) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    // Slow down by 0.5x
    ArrayXr y_slow = time_stretch(y, 0.5);

    // Should be approximately twice the length
    EXPECT_NEAR(static_cast<double>(y_slow.size()), y.size() * 2.0, y.size() * 0.1);
}

TEST(TimeStretchTest, NoChange) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    // Rate = 1.0 should preserve length
    ArrayXr y_same = time_stretch(y, 1.0);

    EXPECT_NEAR(static_cast<double>(y_same.size()), static_cast<double>(y.size()), y.size() * 0.05);
}

TEST(TimeStretchTest, InvalidRate) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    EXPECT_THROW(time_stretch(y, 0.0), ParameterError);
    EXPECT_THROW(time_stretch(y, -1.0), ParameterError);
}

// ============================================================================
// Pitch Shift Tests
// ============================================================================

TEST(PitchShiftTest, BasicShift) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    // Shift up by 12 steps (one octave)
    ArrayXr y_shifted = pitch_shift(y, sr, 12.0);

    // Length should be preserved
    EXPECT_EQ(y_shifted.size(), y.size());
}

TEST(PitchShiftTest, ShiftDown) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    // Shift down by 12 steps (one octave)
    ArrayXr y_shifted = pitch_shift(y, sr, -12.0);

    EXPECT_EQ(y_shifted.size(), y.size());
}

TEST(PitchShiftTest, FractionalSteps) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    // Shift by 2.5 semitones
    ArrayXr y_shifted = pitch_shift(y, sr, 2.5);

    EXPECT_EQ(y_shifted.size(), y.size());
}

TEST(PitchShiftTest, CustomBinsPerOctave) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    // Use 24 bins per octave (quarter tones)
    ArrayXr y_shifted = pitch_shift(y, sr, 3.0, 24);

    EXPECT_EQ(y_shifted.size(), y.size());
}

// ============================================================================
// Trim Tests
// ============================================================================

TEST(TrimTest, BasicTrim) {
    Real sr = 22050;
    ArrayXr y = generate_signal_with_silence(sr, 1.0);

    auto [y_trimmed, indices] = trim(y);

    // Trimmed should be shorter than original
    EXPECT_LT(y_trimmed.size(), y.size());

    // Indices should be valid
    EXPECT_GE(indices.first, 0);
    EXPECT_LE(indices.second, y.size());
    EXPECT_LT(indices.first, indices.second);
}

TEST(TrimTest, NoSilence) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    auto [y_trimmed, indices] = trim(y);

    // Should not trim much from a signal with no silence
    EXPECT_NEAR(static_cast<double>(y_trimmed.size()), static_cast<double>(y.size()), y.size() * 0.1);
}

TEST(TrimTest, AllSilence) {
    ArrayXr y = generate_silence(22050);

    auto [y_trimmed, indices] = trim(y);

    // All silence with ref=max: ref becomes 0, so all frames appear
    // above threshold (matching Python librosa behavior)
    EXPECT_EQ(y_trimmed.size(), y.size());
    EXPECT_EQ(indices.first, 0);
    EXPECT_EQ(indices.second, y.size());
}

TEST(TrimTest, CustomTopDb) {
    Real sr = 22050;
    ArrayXr y = generate_signal_with_silence(sr, 1.0);

    // Lower threshold should trim more aggressively
    auto [y_60db, _1] = trim(y, 60);
    auto [y_30db, _2] = trim(y, 30);

    // Higher threshold should result in shorter or equal output
    EXPECT_LE(y_30db.size(), y_60db.size());
}

// ============================================================================
// Split Tests
// ============================================================================

TEST(SplitTest, BasicSplit) {
    Real sr = 22050;

    // Create signal with silence in the middle
    int n_samples = static_cast<int>(sr * 1.0);
    ArrayXr y(n_samples);

    // First third: sine wave
    for (int i = 0; i < n_samples / 3; ++i) {
        y(i) = std::sin(2 * M_PI * 440.0 * i / sr);
    }
    // Middle third: silence
    for (int i = n_samples / 3; i < 2 * n_samples / 3; ++i) {
        y(i) = 0.0;
    }
    // Last third: sine wave
    for (int i = 2 * n_samples / 3; i < n_samples; ++i) {
        y(i) = std::sin(2 * M_PI * 440.0 * i / sr);
    }

    auto intervals = split(y);

    // Should detect at least one interval
    EXPECT_GE(intervals.size(), 1);

    // Each interval should have valid indices
    for (const auto& interval : intervals) {
        EXPECT_GE(interval.first, 0);
        EXPECT_LE(interval.second, y.size());
        EXPECT_LT(interval.first, interval.second);
    }
}

TEST(SplitTest, AllSilence) {
    ArrayXr y = generate_silence(22050);

    auto intervals = split(y);

    // All silence with ref=max: ref becomes 0, so all frames appear
    // above threshold, returning one interval (matching Python librosa)
    EXPECT_EQ(intervals.size(), 1);
}

TEST(SplitTest, NoSilence) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    auto intervals = split(y);

    // Continuous signal: should be one interval
    EXPECT_EQ(intervals.size(), 1);
}

// ============================================================================
// Preemphasis/Deemphasis Tests
// ============================================================================

TEST(PreemphasisTest, BasicPreemphasis) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.1);

    ArrayXr y_pre = preemphasis(y);

    EXPECT_EQ(y_pre.size(), y.size());

    // First sample should be unchanged
    EXPECT_EQ(y_pre(0), y(0));
}

TEST(PreemphasisTest, DeemphasisInverse) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.1);

    Real coef = 0.97;
    ArrayXr y_pre = preemphasis(y, coef);
    ArrayXr y_de = deemphasis(y_pre, coef);

    // Should approximately recover original signal
    Real max_diff = (y - y_de).abs().maxCoeff();
    EXPECT_LT(max_diff, 1e-10);
}

TEST(PreemphasisTest, CustomCoefficient) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.1);

    ArrayXr y_pre_97 = preemphasis(y, 0.97);
    ArrayXr y_pre_50 = preemphasis(y, 0.50);

    // Different coefficients should give different results
    EXPECT_GT((y_pre_97 - y_pre_50).abs().maxCoeff(), 0);
}

TEST(PreemphasisTest, EmptySignal) {
    ArrayXr y(0);

    ArrayXr y_pre = preemphasis(y);
    ArrayXr y_de = deemphasis(y);

    EXPECT_EQ(y_pre.size(), 0);
    EXPECT_EQ(y_de.size(), 0);
}

// ============================================================================
// Remix Tests
// ============================================================================

TEST(RemixTest, BasicRemix) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    std::vector<std::pair<Eigen::Index, Eigen::Index>> intervals = {
        {0, 5000},
        {10000, 15000}
    };

    ArrayXr y_remixed = remix(y, intervals, false);

    EXPECT_EQ(y_remixed.size(), 10000);
}

TEST(RemixTest, ReverseOrder) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    Eigen::Index mid = y.size() / 2;
    std::vector<std::pair<Eigen::Index, Eigen::Index>> intervals = {
        {mid, y.size()},
        {0, mid}
    };

    ArrayXr y_remixed = remix(y, intervals, false);

    EXPECT_EQ(y_remixed.size(), y.size());
}

TEST(RemixTest, EmptyIntervals) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    std::vector<std::pair<Eigen::Index, Eigen::Index>> intervals;

    ArrayXr y_remixed = remix(y, intervals, false);

    EXPECT_EQ(y_remixed.size(), 0);
}

TEST(RemixTest, AlignZeros) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    std::vector<std::pair<Eigen::Index, Eigen::Index>> intervals = {
        {100, 5000}
    };

    ArrayXr y_no_align = remix(y, intervals, false);
    ArrayXr y_align = remix(y, intervals, true);

    // Both should produce output
    EXPECT_GT(y_no_align.size(), 0);
    EXPECT_GT(y_align.size(), 0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(EffectsTest, ShortSignal) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.05);  // Very short

    // Should handle short signals without crashing
    EXPECT_NO_THROW(time_stretch(y, 1.5));
    EXPECT_NO_THROW(pitch_shift(y, sr, 2.0));
    EXPECT_NO_THROW(preemphasis(y));
}

// ============================================================================
// Harmonic/Percussive Effect Tests
// ============================================================================

TEST(HarmonicTest, BasicHarmonic) {
    Real sr = 22050;
    // Create signal with harmonic content (sine wave) and transient clicks
    int n_samples = static_cast<int>(sr * 0.5);
    ArrayXr y(n_samples);

    // Harmonic: sustained sine wave
    for (int i = 0; i < n_samples; ++i) {
        y(i) = std::sin(2 * M_PI * 440.0 * i / sr);
    }

    // Add transient clicks
    for (int c = 0; c < 4; ++c) {
        int idx = c * n_samples / 4;
        for (int i = 0; i < 100 && idx + i < n_samples; ++i) {
            y(idx + i) += std::exp(-static_cast<Real>(i) / 10.0);
        }
    }

    ArrayXr y_harm = harmonic(y);

    // Output should be same length as input
    EXPECT_EQ(y_harm.size(), y.size());

    // Harmonic component should have less transient energy
    // Compute RMS of first click region vs whole signal
    EXPECT_GT(y_harm.abs().maxCoeff(), 0);
}

TEST(HarmonicTest, DefaultParameters) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    // Should run without errors with default parameters
    EXPECT_NO_THROW(harmonic(y));
}

TEST(PercussiveTest, BasicPercussive) {
    Real sr = 22050;
    int n_samples = static_cast<int>(sr * 0.5);
    ArrayXr y(n_samples);

    // Harmonic: sustained sine wave
    for (int i = 0; i < n_samples; ++i) {
        y(i) = std::sin(2 * M_PI * 440.0 * i / sr);
    }

    // Add transient clicks
    for (int c = 0; c < 4; ++c) {
        int idx = c * n_samples / 4;
        for (int i = 0; i < 100 && idx + i < n_samples; ++i) {
            y(idx + i) += std::exp(-static_cast<Real>(i) / 10.0);
        }
    }

    ArrayXr y_perc = percussive(y);

    // Output should be same length as input
    EXPECT_EQ(y_perc.size(), y.size());

    // Percussive component should have some energy
    EXPECT_GT(y_perc.abs().maxCoeff(), 0);
}

TEST(PercussiveTest, DefaultParameters) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    EXPECT_NO_THROW(percussive(y));
}

TEST(HarmonicPercussiveTest, SumApproximatesOriginal) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);  // Longer signal for reliable interior

    ArrayXr y_harm = harmonic(y);
    ArrayXr y_perc = percussive(y);

    // H + P should approximately equal original in the interior
    // (edges have windowing artifacts with center=true)
    ArrayXr reconstructed = y_harm + y_perc;
    int margin = 2048;  // n_fft
    Eigen::Index interior_len = y.size() - 2 * margin;
    ASSERT_GT(interior_len, 0);
    Real max_diff = (y.segment(margin, interior_len) - reconstructed.segment(margin, interior_len)).abs().maxCoeff();
    Real max_val = y.abs().maxCoeff();

    // Interior should be nearly perfect (HPSS masks sum to 1)
    EXPECT_LT(max_diff / max_val, 0.01);
}
