#include <gtest/gtest.h>
#include <librosa/core/pitch.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>
#include <random>

using namespace librosa;

namespace {

// Random number utility
std::mt19937 rng(628318530);

ArrayXr random_array(Eigen::Index size) {
    std::normal_distribution<Real> dist(0.0, 1.0);
    ArrayXr result(size);
    for (Eigen::Index i = 0; i < size; ++i) {
        result(i) = dist(rng);
    }
    return result;
}

} // anonymous namespace

// ============================================================================
// Pitch Tuning Tests
// ============================================================================

TEST(PitchTuningTest, UntuedFrequencies) {
    // Frequencies at exact semitones (0 tuning offset)
    ArrayXr freqs = cqt_frequencies(24, note_to_hz("C4"), 12, 0.0);

    Real tuning = pitch_tuning(freqs, 0.01, 12);

    // Should be close to 0 (within one bin)
    EXPECT_NEAR(tuning, 0.0, 0.01);
}

TEST(PitchTuningTest, PositiveTuning) {
    // Frequencies at +0.25 bins (quarter-tone sharp)
    ArrayXr freqs = cqt_frequencies(24, note_to_hz("C4"), 12, 0.25);

    Real tuning = pitch_tuning(freqs, 0.01, 12);

    EXPECT_NEAR(tuning, 0.25, 0.02);
}

TEST(PitchTuningTest, NegativeTuning) {
    // Frequencies at -0.25 bins (quarter-tone flat)
    ArrayXr freqs = cqt_frequencies(24, note_to_hz("C4"), 12, -0.25);

    Real tuning = pitch_tuning(freqs, 0.01, 12);

    EXPECT_NEAR(tuning, -0.25, 0.02);
}

TEST(PitchTuningTest, EmptyFrequencies) {
    ArrayXr empty(0);
    Real tuning = pitch_tuning(empty);
    EXPECT_NEAR(tuning, 0.0, 1e-10);
}

TEST(PitchTuningTest, DCFilteredOut) {
    ArrayXr freqs(5);
    freqs << 0.0, 0.0, 440.0, 880.0, 1760.0;

    // Should not crash with DC components
    Real tuning = pitch_tuning(freqs);
    EXPECT_TRUE(std::isfinite(tuning));
}

// ============================================================================
// Piptrack Tests
// ============================================================================

TEST(PiptrackTest, OutputShape) {
    ArrayXr y = random_array(22050);
    int n_fft = 2048;
    int hop_length = 512;

    auto [pitches, mags] = piptrack(y, 22050, n_fft, hop_length);

    // Should have n_fft/2+1 frequency bins
    EXPECT_EQ(pitches.rows(), n_fft / 2 + 1);
    EXPECT_EQ(mags.rows(), n_fft / 2 + 1);

    // Shapes should match
    EXPECT_EQ(pitches.rows(), mags.rows());
    EXPECT_EQ(pitches.cols(), mags.cols());
}

TEST(PiptrackTest, FrequencyBounds) {
    // Generate a tone at known frequency
    Real freq = 440.0;
    Real sr = 22050;
    ArrayXr y = tone(freq, sr, 22050);

    Real fmin = 100.0;
    Real fmax = 1000.0;

    auto [pitches, mags] = piptrack(y, sr, 2048, 512, fmin, fmax);

    // Non-zero pitches should be within bounds
    for (Eigen::Index i = 0; i < pitches.size(); ++i) {
        Real p = pitches.data()[i];
        if (p > 0) {
            EXPECT_GE(p, fmin - 50);  // Allow some tolerance
            EXPECT_LE(p, fmax + 50);
        }
    }
}

TEST(PiptrackTest, PureToneDetection) {
    // Generate a pure tone
    Real freq = 440.0;
    Real sr = 22050;
    ArrayXr y = tone(freq, sr, 22050);

    auto [pitches, mags] = piptrack(y, sr, 2048, 512, 400.0, 500.0, 0.1);

    // Find the most common non-zero pitch
    std::vector<Real> detected;
    for (Eigen::Index i = 0; i < pitches.size(); ++i) {
        if (pitches.data()[i] > 0) {
            detected.push_back(pitches.data()[i]);
        }
    }

    if (!detected.empty()) {
        // Sort and find median
        std::sort(detected.begin(), detected.end());
        Real median_pitch = detected[detected.size() / 2];

        // Should be close to 440 Hz
        EXPECT_NEAR(median_pitch, freq, 20.0);  // Within 20 Hz
    }
}

// ============================================================================
// YIN Tests
// ============================================================================

TEST(YINTest, OutputLength) {
    ArrayXr y = random_array(22050);
    int frame_length = 2048;
    int hop_length = 512;

    ArrayXr f0 = yin(y, 65.0, 2093.0, 22050, frame_length, hop_length);

    // Check output length
    Eigen::Index expected_frames = 1 + (y.size() + frame_length - frame_length) / hop_length;
    EXPECT_NEAR(f0.size(), expected_frames, 2);
}

TEST(YINTest, PureToneEstimation) {
    // Generate a pure tone
    Real freq = 440.0;
    Real sr = 22050;
    Eigen::Index duration_samples = static_cast<Eigen::Index>(sr);

    ArrayXr y = tone(freq, sr, duration_samples);

    ArrayXr f0 = yin(y, 400.0, 500.0, sr, 2048, 512);

    // F0 estimates should be close to 440 Hz
    Real mean_f0 = f0.mean();
    EXPECT_NEAR(mean_f0, freq, 10.0);  // Within 10 Hz
}

TEST(YINTest, ChirpTracking) {
    // Generate a chirp from 440 to 880 Hz
    Real sr = 22050;
    Eigen::Index duration_samples = static_cast<Eigen::Index>(sr * 2);

    ArrayXr y = chirp(440.0, 880.0, sr, duration_samples, std::nullopt, true);

    ArrayXr f0 = yin(y, 400.0, 1000.0, sr, 2048, 512);

    // F0 should generally increase over time
    Eigen::Index quarter = f0.size() / 4;
    Real early_mean = f0.head(quarter).mean();
    Real late_mean = f0.tail(quarter).mean();

    EXPECT_LT(early_mean, late_mean);
}

TEST(YINTest, InvalidParamsThrow) {
    ArrayXr y = random_array(22050);

    // fmax > Nyquist
    EXPECT_THROW(yin(y, 100.0, 20000.0, 22050), ParameterError);

    // fmin >= fmax
    EXPECT_THROW(yin(y, 500.0, 400.0, 22050), ParameterError);

    // fmin <= 0
    EXPECT_THROW(yin(y, -100.0, 500.0, 22050), ParameterError);

    // fmin too small for frame_length
    EXPECT_THROW(yin(y, 10.0, 500.0, 22050, 512), ParameterError);
}

TEST(YINTest, TroughThresholdEffect) {
    ArrayXr y = random_array(22050);

    // Lower threshold should give more stable estimates
    ArrayXr f0_low_thresh = yin(y, 100.0, 500.0, 22050, 2048, 512, 0.01);
    ArrayXr f0_high_thresh = yin(y, 100.0, 500.0, 22050, 2048, 512, 0.5);

    // Both should produce valid output
    EXPECT_EQ(f0_low_thresh.size(), f0_high_thresh.size());
}

// ============================================================================
// Estimate Tuning Tests
// ============================================================================

TEST(EstimateTuningTest, FromAudio) {
    // Generate a tuned tone at A4 (440 Hz)
    ArrayXr y = tone(440.0, 22050, 22050);

    Real tuning = estimate_tuning(y, 22050, 2048, 0.01, 12, 400.0, 500.0, 0.1);

    // Should be close to 0 (standard tuning)
    EXPECT_NEAR(tuning, 0.0, 0.1);
}

TEST(EstimateTuningTest, FromSpectrogramOutput) {
    ArrayXr y = tone(440.0, 22050, 22050);

    Real tuning = estimate_tuning(y, 22050, 2048);

    // Result should be finite
    EXPECT_TRUE(std::isfinite(tuning));
    EXPECT_GE(tuning, -0.5);
    EXPECT_LT(tuning, 0.5);
}

TEST(EstimateTuningTest, ResolutionParameter) {
    ArrayXr freqs = cqt_frequencies(24, note_to_hz("C4"), 12, 0.15);

    Real tuning_coarse = pitch_tuning(freqs, 0.1, 12);
    Real tuning_fine = pitch_tuning(freqs, 0.01, 12);

    // Both should detect positive tuning
    EXPECT_GT(tuning_coarse, 0.0);
    EXPECT_GT(tuning_fine, 0.0);

    // Fine resolution should be more precise
    EXPECT_NEAR(tuning_fine, 0.15, 0.02);
}

// ============================================================================
// pYIN Tests
// ============================================================================

TEST(PyinTest, OutputShape) {
    Real sr = 22050;
    ArrayXr y = random_array(static_cast<Eigen::Index>(sr));
    int frame_length = 2048;
    int hop_length = 512;

    auto result = pyin(y, 65.0, 2093.0, sr, frame_length, hop_length);

    // All three outputs should have the same length
    EXPECT_EQ(result.f0.size(), result.voiced_flag.size());
    EXPECT_EQ(result.f0.size(), result.voiced_prob.size());

    // Check expected number of frames
    Eigen::Index expected_frames = 1 + (y.size() + frame_length - frame_length) / hop_length;
    EXPECT_NEAR(result.f0.size(), expected_frames, 2);
}

TEST(PyinTest, PureTone) {
    Real freq = 440.0;
    Real sr = 22050;
    Eigen::Index duration_samples = static_cast<Eigen::Index>(sr);

    ArrayXr y = tone(freq, sr, duration_samples);

    auto result = pyin(y, 400.0, 500.0, sr, 2048, 512);

    // Count voiced frames
    int voiced_count = 0;
    Real f0_sum = 0.0;
    for (Eigen::Index t = 0; t < result.f0.size(); ++t) {
        if (result.voiced_flag(t)) {
            voiced_count++;
            f0_sum += result.f0(t);
        }
    }

    // Most frames should be voiced
    EXPECT_GT(voiced_count, result.f0.size() / 2);

    // Mean voiced f0 should be close to 440 Hz
    if (voiced_count > 0) {
        Real mean_f0 = f0_sum / voiced_count;
        EXPECT_NEAR(mean_f0, freq, 10.0);
    }
}

TEST(PyinTest, SilentSignal) {
    Real sr = 22050;
    ArrayXr y = ArrayXr::Zero(static_cast<Eigen::Index>(sr));

    auto result = pyin(y, 65.0, 2093.0, sr, 2048, 512);

    // Most frames should be unvoiced
    int unvoiced_count = 0;
    for (Eigen::Index t = 0; t < result.voiced_flag.size(); ++t) {
        if (!result.voiced_flag(t)) {
            unvoiced_count++;
        }
    }
    EXPECT_GT(unvoiced_count, result.voiced_flag.size() / 2);
}

TEST(PyinTest, VoicedProbRange) {
    Real sr = 22050;
    ArrayXr y = random_array(static_cast<Eigen::Index>(sr));

    auto result = pyin(y, 65.0, 2093.0, sr);

    // voiced_prob should be in [0, 1]
    for (Eigen::Index t = 0; t < result.voiced_prob.size(); ++t) {
        EXPECT_GE(result.voiced_prob(t), 0.0);
        EXPECT_LE(result.voiced_prob(t), 1.0);
    }
}

TEST(PyinTest, FrequencyRange) {
    Real sr = 22050;
    Real fmin = 100.0;
    Real fmax = 1000.0;
    ArrayXr y = tone(440.0, sr, static_cast<Eigen::Index>(sr));

    auto result = pyin(y, fmin, fmax, sr, 2048, 512);

    // All voiced f0 values should be between fmin and fmax
    for (Eigen::Index t = 0; t < result.f0.size(); ++t) {
        if (result.voiced_flag(t)) {
            EXPECT_GE(result.f0(t), fmin * 0.95);  // Small tolerance
            EXPECT_LE(result.f0(t), fmax * 1.05);
        }
    }
}

TEST(PyinTest, FillNaDefault) {
    Real sr = 22050;
    ArrayXr y = ArrayXr::Zero(static_cast<Eigen::Index>(sr));

    auto result = pyin(y, 65.0, 2093.0, sr, 2048, 512);

    // Unvoiced frames should have NaN (default fill_na)
    for (Eigen::Index t = 0; t < result.f0.size(); ++t) {
        if (!result.voiced_flag(t)) {
            EXPECT_TRUE(std::isnan(result.f0(t)));
        }
    }
}

TEST(PyinTest, FillNaCustom) {
    Real sr = 22050;
    ArrayXr y = ArrayXr::Zero(static_cast<Eigen::Index>(sr));

    auto result = pyin(y, 65.0, 2093.0, sr, 2048, 512,
                       100, 2.0, 18.0, 2.0, 0.1, 35.92, 0.01, 0.01,
                       0.0);  // fill_na = 0.0

    // Unvoiced frames should have 0.0
    for (Eigen::Index t = 0; t < result.f0.size(); ++t) {
        if (!result.voiced_flag(t)) {
            EXPECT_EQ(result.f0(t), 0.0);
        }
    }
}

TEST(PyinTest, ParameterValidation) {
    ArrayXr y = random_array(22050);

    // fmin >= fmax
    EXPECT_THROW(pyin(y, 500.0, 400.0, 22050), ParameterError);

    // fmin <= 0
    EXPECT_THROW(pyin(y, -100.0, 500.0, 22050), ParameterError);

    // fmax > Nyquist
    EXPECT_THROW(pyin(y, 100.0, 20000.0, 22050), ParameterError);

    // fmin too small for frame_length
    EXPECT_THROW(pyin(y, 10.0, 500.0, 22050, 512), ParameterError);
}
