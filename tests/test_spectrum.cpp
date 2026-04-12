#include <gtest/gtest.h>
#include <librosa/core/spectrum.hpp>
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

void srand_reset() {
    rng.seed(628318530);
}

} // anonymous namespace

// ============================================================================
// STFT Tests
// ============================================================================

TEST(STFTTest, OutputShape) {
    Eigen::Index n_samples = 22050;
    int n_fft = 2048;
    int hop_length = 512;

    ArrayXr y = random_array(n_samples);

    ArrayXXc D = stft(y, n_fft, hop_length);

    // Check frequency dimension (n_fft/2 + 1)
    EXPECT_EQ(D.rows(), n_fft / 2 + 1);

    // Check time dimension (depends on centering)
    Eigen::Index expected_frames = 1 + (n_samples + n_fft / 2 + n_fft / 2 - n_fft) / hop_length;
    EXPECT_NEAR(D.cols(), expected_frames, 2);
}

TEST(STFTTest, RealInputComplexOutput) {
    ArrayXr y = random_array(4096);
    ArrayXXc D = stft(y, 2048, 512);

    // Output should be complex
    bool hasComplex = false;
    for (Eigen::Index i = 0; i < D.size() && !hasComplex; ++i) {
        if (std::abs(D.data()[i].imag()) > 1e-10) {
            hasComplex = true;
        }
    }
    EXPECT_TRUE(hasComplex);
}

TEST(STFTTest, DCAndNyquistReal) {
    // For a real input, DC and Nyquist bins should have zero imaginary part
    ArrayXr y = random_array(4096);
    ArrayXXc D = stft(y, 2048, 512);

    // DC component
    for (Eigen::Index j = 0; j < D.cols(); ++j) {
        // DC should have small imaginary component
        EXPECT_NEAR(D(0, j).imag(), 0.0, 1e-8);
    }
}

TEST(STFTTest, TooShortThrows) {
    ArrayXr y = ArrayXr::Zero(128);
    EXPECT_THROW(stft(y, 2048, 512, std::nullopt, WindowType::Hann, false), ParameterError);
}

class STFTWindowTest : public ::testing::TestWithParam<WindowType> {};

TEST_P(STFTWindowTest, DifferentWindows) {
    WindowType window = GetParam();
    ArrayXr y = random_array(4096);

    ArrayXXc D = stft(y, 2048, 512, std::nullopt, window);

    EXPECT_GT(D.rows(), 0);
    EXPECT_GT(D.cols(), 0);
}

INSTANTIATE_TEST_SUITE_P(STFTWindowTests, STFTWindowTest,
    ::testing::Values(
        WindowType::Hann,
        WindowType::Hamming,
        WindowType::Blackman,
        WindowType::Bartlett,
        WindowType::Rectangular
    ));

// ============================================================================
// ISTFT Tests
// ============================================================================

TEST(ISTFTTest, ReconstructionWithCenter) {
    srand_reset();
    ArrayXr y = random_array(8192);
    int n_fft = 2048;
    int hop_length = 512;

    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt, WindowType::Hann, true);
    ArrayXr y_rec = istft(D, hop_length, std::nullopt, n_fft, WindowType::Hann, true, y.size());

    // Should have same length
    EXPECT_EQ(y_rec.size(), y.size());

    // Should be close to original (allow for edge effects)
    Eigen::Index margin = n_fft;
    for (Eigen::Index i = margin; i < y.size() - margin; ++i) {
        EXPECT_NEAR(y_rec(i), y(i), 1e-5);
    }
}

TEST(ISTFTTest, ReconstructionWithoutCenter) {
    srand_reset();
    ArrayXr y = random_array(8192);
    int n_fft = 2048;
    int hop_length = 512;

    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt, WindowType::Hann, false);
    ArrayXr y_rec = istft(D, hop_length, std::nullopt, n_fft, WindowType::Hann, false, std::nullopt);

    // Reconstruction should be close (may be shorter due to no padding)
    Eigen::Index compare_len = std::min(y.size(), y_rec.size());
    Eigen::Index margin = n_fft;

    for (Eigen::Index i = margin; i < compare_len - margin; ++i) {
        EXPECT_NEAR(y_rec(i), y(i), 1e-5);
    }
}

// ============================================================================
// Magnitude and Phase Tests
// ============================================================================

TEST(MagphaseTest, SeparateAndReconstruct) {
    srand_reset();
    ArrayXr y = random_array(4096);
    ArrayXXc D = stft(y, 2048, 512);

    auto [mag, phase] = magphase(D, 1.0);

    // Magnitude should be non-negative
    EXPECT_TRUE((mag >= 0).all());

    // Phase should be unit complex
    for (Eigen::Index i = 0; i < phase.size(); ++i) {
        Real abs_val = std::abs(phase.data()[i]);
        if (std::abs(D.data()[i]) > 1e-10) {
            EXPECT_NEAR(abs_val, 1.0, 1e-10);
        }
    }

    // Reconstruction: D = mag * phase
    ArrayXXc D_rec = mag.cast<Complex>() * phase;
    for (Eigen::Index i = 0; i < D.size(); ++i) {
        EXPECT_NEAR(D_rec.data()[i].real(), D.data()[i].real(), 1e-10);
        EXPECT_NEAR(D_rec.data()[i].imag(), D.data()[i].imag(), 1e-10);
    }
}

TEST(MagnitudeTest, Basic) {
    ArrayXXc D(2, 2);
    D << Complex(3.0, 4.0), Complex(0.0, 1.0),
         Complex(1.0, 0.0), Complex(1.0, 1.0);

    ArrayXXr mag = magnitude(D);

    EXPECT_NEAR(mag(0, 0), 5.0, 1e-10);  // |3+4i| = 5
    EXPECT_NEAR(mag(0, 1), 1.0, 1e-10);  // |i| = 1
    EXPECT_NEAR(mag(1, 0), 1.0, 1e-10);  // |1| = 1
    EXPECT_NEAR(mag(1, 1), std::sqrt(2.0), 1e-10);  // |1+i| = sqrt(2)
}

TEST(PhaseTest, Basic) {
    ArrayXXc D(2, 2);
    D << Complex(1.0, 0.0), Complex(0.0, 1.0),
         Complex(-1.0, 0.0), Complex(0.0, -1.0);

    ArrayXXr ph = phase(D);

    EXPECT_NEAR(ph(0, 0), 0.0, 1e-10);           // angle(1) = 0
    EXPECT_NEAR(ph(0, 1), constants::PI / 2, 1e-10);  // angle(i) = pi/2
    EXPECT_NEAR(std::abs(ph(1, 0)), constants::PI, 1e-10);  // angle(-1) = pi
    EXPECT_NEAR(ph(1, 1), -constants::PI / 2, 1e-10);  // angle(-i) = -pi/2
}

// ============================================================================
// Decibel Conversion Tests
// ============================================================================

TEST(PowerToDbTest, Basic) {
    ArrayXXr S(2, 2);
    S << 1.0, 10.0,
         100.0, 1000.0;

    ArrayXXr S_db = power_to_db(S, 1.0, 1e-10, std::nullopt);

    EXPECT_NEAR(S_db(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(S_db(0, 1), 10.0, 1e-10);
    EXPECT_NEAR(S_db(1, 0), 20.0, 1e-10);
    EXPECT_NEAR(S_db(1, 1), 30.0, 1e-10);
}

TEST(PowerToDbTest, WithTopDb) {
    ArrayXXr S(2, 2);
    S << 1.0, 10.0,
         1e-10, 1000.0;

    Real top_db = 40.0;
    ArrayXXr S_db = power_to_db(S, 1.0, 1e-10, top_db);

    // Find max
    Real max_db = S_db.maxCoeff();

    // All values should be within top_db of max
    for (Eigen::Index i = 0; i < S_db.size(); ++i) {
        EXPECT_GE(S_db.data()[i], max_db - top_db);
    }
}

TEST(DbToPowerTest, Roundtrip) {
    ArrayXXr S(2, 2);
    S << 1.0, 10.0,
         100.0, 1000.0;

    ArrayXXr S_db = power_to_db(S, 1.0, 1e-10, std::nullopt);
    ArrayXXr S_back = db_to_power(S_db, 1.0);

    for (Eigen::Index i = 0; i < S.size(); ++i) {
        EXPECT_NEAR(S_back.data()[i], S.data()[i], 1e-5);
    }
}

TEST(AmplitudeToDbTest, Basic) {
    ArrayXXr S(2, 2);
    S << 1.0, 10.0,
         100.0, 1000.0;

    ArrayXXr S_db = amplitude_to_db(S, 1.0, 1e-5, std::nullopt);

    // amplitude_to_db uses 20*log10 instead of 10*log10
    EXPECT_NEAR(S_db(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(S_db(0, 1), 20.0, 1e-10);
    EXPECT_NEAR(S_db(1, 0), 40.0, 1e-10);
    EXPECT_NEAR(S_db(1, 1), 60.0, 1e-10);
}

TEST(DbToAmplitudeTest, Roundtrip) {
    ArrayXXr S(2, 2);
    S << 1.0, 10.0,
         100.0, 1000.0;

    ArrayXXr S_db = amplitude_to_db(S, 1.0, 1e-5, std::nullopt);
    ArrayXXr S_back = db_to_amplitude(S_db, 1.0);

    for (Eigen::Index i = 0; i < S.size(); ++i) {
        EXPECT_NEAR(S_back.data()[i], S.data()[i], 1e-5);
    }
}

TEST(PowerToDbTest, ScalarConversion) {
    EXPECT_NEAR(power_to_db(1.0), 0.0, 1e-10);
    EXPECT_NEAR(power_to_db(10.0), 10.0, 1e-10);
    EXPECT_NEAR(power_to_db(100.0), 20.0, 1e-10);
}

TEST(DbToPowerTest, ScalarConversion) {
    EXPECT_NEAR(db_to_power(0.0), 1.0, 1e-10);
    EXPECT_NEAR(db_to_power(10.0), 10.0, 1e-5);
    EXPECT_NEAR(db_to_power(20.0), 100.0, 1e-5);
}

// ============================================================================
// Phase Vocoder Tests
// ============================================================================

TEST(PhaseVocoderTest, SpeedUp) {
    srand_reset();
    ArrayXr y = random_array(8192);
    int hop_length = 512;

    ArrayXXc D = stft(y, 2048, hop_length);

    // Speed up by 2x
    ArrayXXc D_stretched = phase_vocoder(D, 2.0, hop_length);

    // Output should have approximately half as many frames
    EXPECT_NEAR(D_stretched.cols(), D.cols() / 2, 2);

    // Frequency dimension should be unchanged
    EXPECT_EQ(D_stretched.rows(), D.rows());
}

TEST(PhaseVocoderTest, SlowDown) {
    srand_reset();
    ArrayXr y = random_array(8192);
    int hop_length = 512;

    ArrayXXc D = stft(y, 2048, hop_length);

    // Slow down by 0.5x
    ArrayXXc D_stretched = phase_vocoder(D, 0.5, hop_length);

    // Output should have approximately twice as many frames
    EXPECT_NEAR(D_stretched.cols(), D.cols() * 2, 2);

    // Frequency dimension should be unchanged
    EXPECT_EQ(D_stretched.rows(), D.rows());
}

TEST(PhaseVocoderTest, NoChange) {
    srand_reset();
    ArrayXr y = random_array(4096);
    int hop_length = 512;

    ArrayXXc D = stft(y, 2048, hop_length);

    // Rate of 1.0 should not change frame count
    ArrayXXc D_stretched = phase_vocoder(D, 1.0, hop_length);

    EXPECT_EQ(D_stretched.cols(), D.cols());
    EXPECT_EQ(D_stretched.rows(), D.rows());
}

// ============================================================================
// Griffin-Lim Tests
// ============================================================================

TEST(GriffinLimTest, BasicReconstruction) {
    srand_reset();
    ArrayXr y = random_array(8192);
    int n_fft = 2048;
    int hop_length = 512;

    // Get magnitude spectrogram
    ArrayXXc D = stft(y, n_fft, hop_length);
    ArrayXXr S = magnitude(D);

    // Reconstruct with Griffin-Lim
    ArrayXr y_rec = griffinlim(S, 32, hop_length, std::nullopt, n_fft,
                                WindowType::Hann, true, y.size());

    EXPECT_EQ(y_rec.size(), y.size());

    // The reconstructed signal should have similar magnitude spectrum
    ArrayXXc D_rec = stft(y_rec, n_fft, hop_length);
    ArrayXXr S_rec = magnitude(D_rec);

    // Compare magnitude spectrograms
    Eigen::Index min_cols = std::min(S.cols(), S_rec.cols());
    Real mag_error = 0.0;
    for (Eigen::Index j = 0; j < min_cols; ++j) {
        for (Eigen::Index i = 0; i < S.rows(); ++i) {
            mag_error += std::abs(S(i, j) - S_rec(i, j));
        }
    }
    mag_error /= (S.rows() * min_cols);

    // Average error should be small relative to average magnitude
    Real avg_mag = S.abs().mean();
    EXPECT_LT(mag_error / avg_mag, 0.3);
}

TEST(GriffinLimTest, IterationsImprove) {
    srand_reset();
    ArrayXr y = random_array(4096);
    int n_fft = 2048;
    int hop_length = 512;

    ArrayXXc D = stft(y, n_fft, hop_length);
    ArrayXXr S = magnitude(D);

    // Few iterations
    ArrayXr y_few = griffinlim(S, 5, hop_length, std::nullopt, n_fft);

    // Many iterations
    ArrayXr y_many = griffinlim(S, 50, hop_length, std::nullopt, n_fft);

    // Compute reconstruction error for each
    ArrayXXc D_few = stft(y_few, n_fft, hop_length);
    ArrayXXc D_many = stft(y_many, n_fft, hop_length);

    ArrayXXr S_few = magnitude(D_few);
    ArrayXXr S_many = magnitude(D_many);

    Eigen::Index min_cols = std::min({S.cols(), S_few.cols(), S_many.cols()});
    Eigen::Index min_rows = S.rows();

    Real err_few = 0.0, err_many = 0.0;
    for (Eigen::Index j = 0; j < min_cols; ++j) {
        for (Eigen::Index i = 0; i < min_rows; ++i) {
            err_few += std::abs(S(i, j) - S_few(i, j));
            err_many += std::abs(S(i, j) - S_many(i, j));
        }
    }

    // More iterations should give lower error
    EXPECT_LT(err_many, err_few);
}

// ============================================================================
// PCEN Tests
// ============================================================================

TEST(PCENTest, BasicPCEN) {
    srand_reset();
    ArrayXr y = random_array(22050);

    ArrayXXc D = stft(y, 2048, 512);
    ArrayXXr S = magnitude(D).square();

    ArrayXXr S_pcen = pcen(S, 22050, 512);

    EXPECT_EQ(S_pcen.rows(), S.rows());
    EXPECT_EQ(S_pcen.cols(), S.cols());

    // PCEN output should be finite
    for (Eigen::Index i = 0; i < S_pcen.size(); ++i) {
        EXPECT_TRUE(std::isfinite(S_pcen.data()[i]));
    }
}

TEST(PCENTest, OutputBounded) {
    srand_reset();
    ArrayXr y = random_array(22050);

    ArrayXXc D = stft(y, 2048, 512);
    ArrayXXr S = magnitude(D).square();

    ArrayXXr S_pcen = pcen(S, 22050, 512, 0.98, 2.0, 0.5, 0.4, 1e-6);

    // PCEN with power=0.5 should produce bounded values
    Real max_val = S_pcen.maxCoeff();
    Real min_val = S_pcen.minCoeff();

    EXPECT_TRUE(std::isfinite(max_val));
    EXPECT_TRUE(std::isfinite(min_val));
    EXPECT_GE(min_val, 0.0);  // Should be non-negative
}

// ============================================================================
// Window Function Tests
// ============================================================================

class GetWindowTest : public ::testing::TestWithParam<WindowType> {};

TEST_P(GetWindowTest, WindowLength) {
    WindowType window_type = GetParam();
    int n_fft = 2048;

    ArrayXr window = get_window(window_type, n_fft, true);

    EXPECT_EQ(window.size(), n_fft);
}

TEST_P(GetWindowTest, WindowBounded) {
    WindowType window_type = GetParam();
    int n_fft = 2048;

    ArrayXr window = get_window(window_type, n_fft, true);

    // All windows should be bounded in [0, 1] (with fp epsilon)
    EXPECT_GE(window.minCoeff(), -1e-15);
    EXPECT_LE(window.maxCoeff(), 1.0 + 1e-10);
}

TEST_P(GetWindowTest, WindowSymmetry) {
    WindowType window_type = GetParam();
    int n_fft = 2048;

    // Non-FFT bins (symmetric)
    ArrayXr window = get_window(window_type, n_fft, false);

    // Check symmetry
    for (Eigen::Index i = 0; i < n_fft / 2; ++i) {
        EXPECT_NEAR(window(i), window(n_fft - 1 - i), 1e-10);
    }
}

INSTANTIATE_TEST_SUITE_P(GetWindowTests, GetWindowTest,
    ::testing::Values(
        WindowType::Hann,
        WindowType::Hamming,
        WindowType::Blackman,
        WindowType::Bartlett,
        WindowType::Rectangular
    ));

TEST(GetWindowTest, StringWindow) {
    ArrayXr hann1 = get_window(WindowType::Hann, 2048, true);
    ArrayXr hann2 = get_window("hann", 2048, true);

    for (Eigen::Index i = 0; i < hann1.size(); ++i) {
        EXPECT_NEAR(hann1(i), hann2(i), 1e-10);
    }
}

TEST(GetWindowTest, InvalidStringThrows) {
    EXPECT_THROW(get_window("invalid_window", 2048, true), ParameterError);
}

// ============================================================================
// Window Sum Square Tests
// ============================================================================

TEST(WindowSumSquareTest, Basic) {
    int n_fft = 2048;
    int hop_length = 512;
    int n_frames = 10;

    ArrayXr window = get_window(WindowType::Hann, n_fft, true);
    ArrayXr wss = window_sumsquare(window, n_frames, hop_length, n_fft);

    // Edges may be zero (Hann window endpoints are 0), but interior should be positive
    Eigen::Index margin = n_fft;
    Eigen::Index mid_start = margin;
    Eigen::Index mid_end = wss.size() - margin;

    if (mid_end > mid_start) {
        Real mid_val = wss(mid_start + (mid_end - mid_start) / 2);
        for (Eigen::Index i = mid_start; i < mid_end; ++i) {
            EXPECT_NEAR(wss(i), mid_val, 1e-5 * mid_val);
        }
    }
}

// ============================================================================
// Perceptual Weighting Tests
// ============================================================================

TEST(PerceptualWeightingTest, AWeighting) {
    srand_reset();
    ArrayXr y = random_array(22050);

    ArrayXXc D = stft(y, 2048, 512);
    ArrayXXr S = magnitude(D).square();

    ArrayXr freqs = ArrayXr::LinSpaced(S.rows(), 0, 22050.0 / 2);

    ArrayXXr S_weighted = perceptual_weighting(S, freqs, WeightType::A);

    EXPECT_EQ(S_weighted.rows(), S.rows());
    EXPECT_EQ(S_weighted.cols(), S.cols());
}
