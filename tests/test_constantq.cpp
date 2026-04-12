#include <gtest/gtest.h>
#include <librosa/core/constantq.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;

namespace {

// Generate a pure tone
ArrayXr make_tone(Real freq, Real sr, int n_samples) {
    ArrayXr t = ArrayXr::LinSpaced(n_samples, 0, (n_samples - 1) / sr);
    return (2.0 * constants::PI * freq * t).cos();
}

} // namespace

// ============================================================================
// CQT Shape Tests
// ============================================================================

TEST(CQT, OutputShape_Default) {
    // Default parameters: 84 bins, 12 bins/octave
    ArrayXr y = make_tone(440.0, 22050, 22050);
    ArrayXXc C = cqt(y, 22050, 512);

    EXPECT_EQ(C.rows(), 84);
    EXPECT_GT(C.cols(), 0);
}

TEST(CQT, OutputShape_CustomBins) {
    ArrayXr y = make_tone(440.0, 22050, 22050);
    ArrayXXc C = cqt(y, 22050, 512, std::nullopt, 60, 12);

    EXPECT_EQ(C.rows(), 60);
    EXPECT_GT(C.cols(), 0);
}

TEST(CQT, OutputShape_HigherResolution) {
    ArrayXr y = make_tone(440.0, 22050, 22050);
    ArrayXXc C = cqt(y, 22050, 512, std::nullopt, 48, 24);

    EXPECT_EQ(C.rows(), 48);
    EXPECT_GT(C.cols(), 0);
}

// ============================================================================
// CQT Frequency Test
// ============================================================================

TEST(CQT, PureTonePeaksNearA4) {
    // A 440 Hz tone should peak near the A4 bin
    Real sr = 22050;
    int n_samples = 2 * static_cast<int>(sr);  // 2 seconds
    ArrayXr y = make_tone(440.0, sr, n_samples);

    int n_bins = 84;
    int bins_per_octave = 12;
    Real fmin = note_to_hz("C1");

    ArrayXXc C = cqt(y, sr, 512, fmin, n_bins, bins_per_octave, 0.0);

    // Compute CQT frequencies
    ArrayXr freqs = cqt_frequencies(n_bins, fmin, bins_per_octave);

    // Find the bin closest to 440 Hz
    Eigen::Index expected_bin = 0;
    Real min_diff = std::abs(freqs(0) - 440.0);
    for (Eigen::Index i = 1; i < freqs.size(); ++i) {
        Real diff = std::abs(freqs(i) - 440.0);
        if (diff < min_diff) {
            min_diff = diff;
            expected_bin = i;
        }
    }

    // Find the bin with maximum average magnitude
    ArrayXr avg_mag(n_bins);
    for (int i = 0; i < n_bins; ++i) {
        avg_mag(i) = C.row(i).abs().mean();
    }

    Eigen::Index peak_bin;
    avg_mag.maxCoeff(&peak_bin);

    // Peak should be within 2 bins of expected
    EXPECT_NEAR(static_cast<double>(peak_bin), static_cast<double>(expected_bin), 2.0);
}

// ============================================================================
// CQT == VQT with gamma=0
// ============================================================================

TEST(CQT, EqualsVQTWithGammaZero) {
    ArrayXr y = make_tone(440.0, 22050, 22050);

    ArrayXXc C_cqt = cqt(y, 22050, 512, std::nullopt, 84, 12, 0.0);
    ArrayXXc C_vqt = vqt(y, 22050, 512, std::nullopt, 84, 0.0, 12, 0.0);

    ASSERT_EQ(C_cqt.rows(), C_vqt.rows());
    ASSERT_EQ(C_cqt.cols(), C_vqt.cols());

    // Compare magnitudes
    ArrayXXr mag_cqt = C_cqt.abs();
    ArrayXXr mag_vqt = C_vqt.abs();

    Real max_diff = (mag_cqt - mag_vqt).abs().maxCoeff();
    Real max_val = std::max(mag_cqt.maxCoeff(), mag_vqt.maxCoeff());

    EXPECT_LT(max_diff / (max_val + 1e-10), 1e-6);
}

// ============================================================================
// VQT Tests
// ============================================================================

TEST(VQT, OutputShape) {
    ArrayXr y = make_tone(440.0, 22050, 22050);
    ArrayXXc V = vqt(y, 22050, 512, std::nullopt, 84);

    EXPECT_EQ(V.rows(), 84);
    EXPECT_GT(V.cols(), 0);
}

TEST(VQT, WithERBGamma) {
    // gamma=nullopt triggers ERB-proportional bandwidth
    ArrayXr y = make_tone(440.0, 22050, 22050);
    ArrayXXc V = vqt(y, 22050, 512, std::nullopt, 84, std::nullopt);

    EXPECT_EQ(V.rows(), 84);
    EXPECT_GT(V.cols(), 0);
}

// ============================================================================
// Pseudo CQT Tests
// ============================================================================

TEST(PseudoCQT, ReturnsRealOutput) {
    ArrayXr y = make_tone(440.0, 22050, 22050);
    ArrayXXr C = pseudo_cqt(y, 22050, 512, std::nullopt, 84);

    EXPECT_EQ(C.rows(), 84);
    EXPECT_GT(C.cols(), 0);

    // All values should be non-negative (it's magnitude)
    EXPECT_GE(C.minCoeff(), 0.0);
}

TEST(PseudoCQT, OutputShape) {
    ArrayXr y = make_tone(440.0, 22050, 22050);
    ArrayXXr C = pseudo_cqt(y, 22050, 512, std::nullopt, 60, 12);

    EXPECT_EQ(C.rows(), 60);
    EXPECT_GT(C.cols(), 0);
}

// ============================================================================
// ICQT Round-trip Tests
// ============================================================================

TEST(ICQT, RoundTripReconstruction) {
    Real sr = 22050;
    int n_samples = static_cast<int>(sr);  // 1 second
    ArrayXr y = make_tone(440.0, sr, n_samples);

    int n_bins = 84;
    int hop_length = 512;

    ArrayXXc C = cqt(y, sr, hop_length, std::nullopt, n_bins, 12, 0.0);
    ArrayXr y_hat = icqt(C, sr, hop_length, std::nullopt, 12, 0.0);

    // Trim to same length
    Eigen::Index min_len = std::min(y.size(), y_hat.size());
    // Skip edges which have windowing artifacts
    Eigen::Index margin = 2048;
    if (min_len > 2 * margin) {
        ArrayXr y_trim = y.segment(margin, min_len - 2 * margin);
        ArrayXr yh_trim = y_hat.segment(margin, min_len - 2 * margin);

        // Compute normalized correlation
        Real y_mean = y_trim.mean();
        Real yh_mean = yh_trim.mean();
        ArrayXr y_centered = y_trim - y_mean;
        ArrayXr yh_centered = yh_trim - yh_mean;

        Real num = (y_centered * yh_centered).sum();
        Real den = std::sqrt((y_centered * y_centered).sum() * (yh_centered * yh_centered).sum());
        Real corr = (den > 0) ? num / den : 0.0;

        EXPECT_GT(corr, 0.8) << "Round-trip reconstruction correlation too low";
    }
}

// ============================================================================
// Griffin-Lim CQT Tests
// ============================================================================

TEST(GriffinLimCQT, BasicReconstruction) {
    Real sr = 22050;
    int n_samples = static_cast<int>(sr);  // 1 second
    ArrayXr y = make_tone(440.0, sr, n_samples);

    int n_bins = 36;  // Fewer bins for speed
    int hop_length = 512;

    ArrayXXc C = cqt(y, sr, hop_length, std::nullopt, n_bins, 12, 0.0);
    ArrayXXr C_mag = C.abs();

    // Just a few iterations for test speed
    ArrayXr y_hat = griffinlim_cqt(C_mag, 4, sr, hop_length,
                                    std::nullopt, 12, 0.0, 1.0,
                                    1.0, 0.01, WindowType::Hann, true,
                                    PadMode::Constant, std::nullopt, 0.99,
                                    "random", 42);

    // Just check it produces output of reasonable length
    EXPECT_GT(y_hat.size(), 0);
}

// ============================================================================
// Parameter Validation Tests
// ============================================================================

TEST(CQT, FminTooHigh) {
    ArrayXr y = make_tone(440.0, 22050, 22050);
    // fmin = 20000 Hz with 84 bins would exceed Nyquist
    EXPECT_THROW(cqt(y, 22050, 512, 20000.0, 84), ParameterError);
}

TEST(CQT, ShortSignal) {
    // Very short signal
    ArrayXr y(10);
    y.setOnes();
    // This should either work or throw a reasonable error
    // depending on the implementation
    EXPECT_NO_THROW({
        try {
            cqt(y, 22050, 512, std::nullopt, 12, 12, 0.0);
        } catch (const ParameterError&) {
            // Expected for very short signals
        }
    });
}

// ============================================================================
// Early Downsample Tests
// ============================================================================

TEST(CQT, WorksWithSmallerHopLength) {
    ArrayXr y = make_tone(440.0, 22050, 22050);
    ArrayXXc C = cqt(y, 22050, 256, std::nullopt, 84, 12, 0.0);

    EXPECT_EQ(C.rows(), 84);
    // With hop_length=256, should have roughly 2x more frames than hop=512
    EXPECT_GT(C.cols(), 0);
}

TEST(CQT, WorksWithFewBins) {
    ArrayXr y = make_tone(440.0, 22050, 22050);
    ArrayXXc C = cqt(y, 22050, 512, std::nullopt, 12, 12, 0.0);

    EXPECT_EQ(C.rows(), 12);
    EXPECT_GT(C.cols(), 0);
}
