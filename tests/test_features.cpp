#include <gtest/gtest.h>
#include <librosa/feature/spectral.hpp>
#include <librosa/feature/utils.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/core/constantq.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;
using namespace librosa::feature;

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

ArrayXr generate_white_noise(int n_samples, unsigned int seed = 42) {
    std::srand(seed);
    ArrayXr y(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        y(i) = 2.0 * std::rand() / RAND_MAX - 1.0;
    }
    return y;
}

} // anonymous namespace

// ============================================================================
// Mel Spectrogram Tests
// ============================================================================

TEST(MelSpectrogramTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr S = melspectrogram(y, sr);

    EXPECT_GT(S.rows(), 0);
    EXPECT_GT(S.cols(), 0);
    EXPECT_EQ(S.rows(), 128);  // Default n_mels
    EXPECT_TRUE((S >= 0).all());  // Non-negative energies
}

TEST(MelSpectrogramTest, CustomParameters) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    int n_mels = 64;
    int n_fft = 1024;
    int hop_length = 256;

    ArrayXXr S = melspectrogram(y, sr, n_fft, hop_length, std::nullopt,
                                WindowType::Hann, true, PadMode::Constant,
                                2.0, n_mels);

    EXPECT_EQ(S.rows(), n_mels);
    EXPECT_TRUE((S >= 0).all());
}

TEST(MelSpectrogramTest, HTKFormula) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr S_slaney = melspectrogram(y, sr, 2048, 512, std::nullopt,
                                       WindowType::Hann, true, PadMode::Constant,
                                       2.0, 128, 0.0, std::nullopt, false);
    ArrayXXr S_htk = melspectrogram(y, sr, 2048, 512, std::nullopt,
                                    WindowType::Hann, true, PadMode::Constant,
                                    2.0, 128, 0.0, std::nullopt, true);

    // HTK and Slaney should produce different results
    EXPECT_GT((S_slaney - S_htk).abs().maxCoeff(), 0);
}

// ============================================================================
// MFCC Tests
// ============================================================================

TEST(MFCCTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr M = mfcc(y, sr);

    EXPECT_EQ(M.rows(), 20);  // Default n_mfcc
    EXPECT_GT(M.cols(), 0);
}

TEST(MFCCTest, CustomNumberOfCoefficients) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    int n_mfcc = 13;
    ArrayXXr M = mfcc(y, sr, n_mfcc);

    EXPECT_EQ(M.rows(), n_mfcc);
}

TEST(MFCCTest, DCTTypes) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr M2 = mfcc(y, sr, 20, 2);  // DCT-II
    ArrayXXr M3 = mfcc(y, sr, 20, 3);  // DCT-III

    // DCT-II and DCT-III should produce different results
    EXPECT_GT((M2 - M3).abs().maxCoeff(), 0);
}

TEST(MFCCTest, Liftering) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr M_no_lift = mfcc(y, sr, 20, 2, true, 0);
    ArrayXXr M_lift = mfcc(y, sr, 20, 2, true, 22);

    // Liftering should modify the result
    EXPECT_GT((M_no_lift - M_lift).abs().maxCoeff(), 0);
}

TEST(MFCCTest, FromLogMelSpectrogram) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    // Compute mel spectrogram and convert to dB
    ArrayXXr S_mel = melspectrogram(y, sr);
    ArrayXXr S_db = power_to_db(S_mel);

    // Compute MFCCs from log-mel spectrogram
    ArrayXXr M = mfcc(S_db, 20);

    EXPECT_EQ(M.rows(), 20);
}

// ============================================================================
// Chroma Tests
// ============================================================================

TEST(ChromaTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);  // A4

    ArrayXXr chroma = chroma_stft(y, sr);

    EXPECT_EQ(chroma.rows(), 12);  // 12 chroma bins
    EXPECT_GT(chroma.cols(), 0);
}

TEST(ChromaTest, PitchDetection) {
    Real sr = 22050;
    // Generate A4 (440 Hz) - should have energy in chroma bin 9 (A)
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    // Use explicit parameters to avoid ambiguity
    ArrayXXr chroma = chroma_stft(y, sr, 2048, 512, 12, std::optional<Real>(0.0));

    // Check that some energy is present
    EXPECT_GT(chroma.maxCoeff(), 0);
}

TEST(ChromaTest, CustomChromaBins) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    int n_chroma = 24;
    ArrayXXr chroma = chroma_stft(y, sr, 2048, 512, n_chroma);

    EXPECT_EQ(chroma.rows(), n_chroma);
}

// ============================================================================
// Spectral Centroid Tests
// ============================================================================

TEST(SpectralCentroidTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr centroid = spectral_centroid(y, sr);

    EXPECT_EQ(centroid.rows(), 1);
    EXPECT_GT(centroid.cols(), 0);
    EXPECT_TRUE((centroid >= 0).all());
}

TEST(SpectralCentroidTest, FrequencyDetection) {
    Real sr = 22050;
    Real freq = 1000.0;
    ArrayXr y = generate_sine(freq, sr, 0.5);

    ArrayXXr centroid = spectral_centroid(y, sr);

    // Mean centroid should be close to the signal frequency
    Real mean_centroid = centroid.mean();
    EXPECT_NEAR(mean_centroid, freq, 200);  // Within 200 Hz
}

TEST(SpectralCentroidTest, HigherFrequencyHigherCentroid) {
    Real sr = 22050;

    ArrayXr y_low = generate_sine(500.0, sr, 0.5);
    ArrayXr y_high = generate_sine(2000.0, sr, 0.5);

    ArrayXXr centroid_low = spectral_centroid(y_low, sr);
    ArrayXXr centroid_high = spectral_centroid(y_high, sr);

    EXPECT_LT(centroid_low.mean(), centroid_high.mean());
}

// ============================================================================
// Spectral Bandwidth Tests
// ============================================================================

TEST(SpectralBandwidthTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr bw = spectral_bandwidth(y, sr);

    EXPECT_EQ(bw.rows(), 1);
    EXPECT_GT(bw.cols(), 0);
    EXPECT_TRUE((bw >= 0).all());
}

TEST(SpectralBandwidthTest, PureToneNarrowBandwidth) {
    Real sr = 22050;

    // Pure tone should have narrow bandwidth
    ArrayXr y_sine = generate_sine(1000.0, sr, 0.5);
    // Noise should have wider bandwidth
    ArrayXr y_noise = generate_white_noise(static_cast<int>(sr * 0.5));

    ArrayXXr bw_sine = spectral_bandwidth(y_sine, sr);
    ArrayXXr bw_noise = spectral_bandwidth(y_noise, sr);

    EXPECT_LT(bw_sine.mean(), bw_noise.mean());
}

TEST(SpectralBandwidthTest, DifferentPValues) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr bw_p1 = spectral_bandwidth(y, sr, 2048, 512, WindowType::Hann, true, 1);
    ArrayXXr bw_p2 = spectral_bandwidth(y, sr, 2048, 512, WindowType::Hann, true, 2);

    // Different p values should produce different results
    EXPECT_GT((bw_p1 - bw_p2).abs().maxCoeff(), 0);
}

// ============================================================================
// Spectral Rolloff Tests
// ============================================================================

TEST(SpectralRolloffTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr rolloff = spectral_rolloff(y, sr);

    EXPECT_EQ(rolloff.rows(), 1);
    EXPECT_GT(rolloff.cols(), 0);
    EXPECT_TRUE((rolloff >= 0).all());
    EXPECT_TRUE((rolloff <= sr / 2).all());  // Below Nyquist
}

TEST(SpectralRolloffTest, DifferentRollPercent) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr rolloff_85 = spectral_rolloff(y, sr, 2048, 512, WindowType::Hann, true, 0.85);
    ArrayXXr rolloff_95 = spectral_rolloff(y, sr, 2048, 512, WindowType::Hann, true, 0.95);

    // Higher percentage should give higher rolloff frequency
    EXPECT_LE(rolloff_85.mean(), rolloff_95.mean());
}

TEST(SpectralRolloffTest, InvalidRollPercent) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    EXPECT_THROW(spectral_rolloff(y, sr, 2048, 512, WindowType::Hann, true, 0.0), ParameterError);
    EXPECT_THROW(spectral_rolloff(y, sr, 2048, 512, WindowType::Hann, true, 1.0), ParameterError);
}

// ============================================================================
// Spectral Flatness Tests
// ============================================================================

TEST(SpectralFlatnessTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr flatness = spectral_flatness(y);

    EXPECT_EQ(flatness.rows(), 1);
    EXPECT_GT(flatness.cols(), 0);
    EXPECT_TRUE((flatness >= 0).all());
    EXPECT_TRUE((flatness <= 1).all());
}

TEST(SpectralFlatnessTest, ToneVsNoise) {
    Real sr = 22050;

    // Pure tone should have low flatness
    ArrayXr y_sine = generate_sine(440.0, sr, 0.5);
    // White noise should have high flatness
    ArrayXr y_noise = generate_white_noise(static_cast<int>(sr * 0.5));

    ArrayXXr flatness_sine = spectral_flatness(y_sine);
    ArrayXXr flatness_noise = spectral_flatness(y_noise);

    EXPECT_LT(flatness_sine.mean(), flatness_noise.mean());
}

TEST(SpectralFlatnessTest, InvalidAmin) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    EXPECT_THROW(spectral_flatness(y, 2048, 512, WindowType::Hann, true, 0.0), ParameterError);
    EXPECT_THROW(spectral_flatness(y, 2048, 512, WindowType::Hann, true, -1.0), ParameterError);
}

// ============================================================================
// Spectral Contrast Tests
// ============================================================================

TEST(SpectralContrastTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr contrast = spectral_contrast(y, sr);

    EXPECT_EQ(contrast.rows(), 7);  // 6 bands + 1
    EXPECT_GT(contrast.cols(), 0);
}

TEST(SpectralContrastTest, CustomBands) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    int n_bands = 4;
    ArrayXXr contrast = spectral_contrast(y, sr, 2048, 512, WindowType::Hann, true, 200.0, n_bands);

    EXPECT_EQ(contrast.rows(), n_bands + 1);
}

TEST(SpectralContrastTest, LinearVsLog) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr contrast_log = spectral_contrast(y, sr, 2048, 512, WindowType::Hann, true, 200.0, 6, 0.02, false);
    ArrayXXr contrast_lin = spectral_contrast(y, sr, 2048, 512, WindowType::Hann, true, 200.0, 6, 0.02, true);

    // Linear and log should produce different results
    EXPECT_GT((contrast_log - contrast_lin).abs().maxCoeff(), 0);
}

// ============================================================================
// RMS Tests
// ============================================================================

TEST(RMSTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr energy = rms(y);

    EXPECT_EQ(energy.rows(), 1);
    EXPECT_GT(energy.cols(), 0);
    EXPECT_TRUE((energy >= 0).all());
}

TEST(RMSTest, UnitAmplitudeSine) {
    Real sr = 22050;
    // Unit amplitude sine wave has RMS = 1/sqrt(2) ≈ 0.707
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr energy = rms(y);
    Real mean_rms = energy.mean();

    EXPECT_NEAR(mean_rms, 1.0 / std::sqrt(2.0), 0.1);
}

TEST(RMSTest, LouderIsHigherRMS) {
    Real sr = 22050;

    ArrayXr y_quiet = generate_sine(440.0, sr, 0.5) * 0.5;
    ArrayXr y_loud = generate_sine(440.0, sr, 0.5);

    ArrayXXr rms_quiet = rms(y_quiet);
    ArrayXXr rms_loud = rms(y_loud);

    EXPECT_LT(rms_quiet.mean(), rms_loud.mean());
}

// ============================================================================
// Zero Crossing Rate Tests
// ============================================================================

TEST(ZeroCrossingRateTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr zcr = zero_crossing_rate(y);

    EXPECT_EQ(zcr.rows(), 1);
    EXPECT_GT(zcr.cols(), 0);
    EXPECT_TRUE((zcr >= 0).all());
    EXPECT_TRUE((zcr <= 1).all());  // ZCR is a rate
}

TEST(ZeroCrossingRateTest, HigherFrequencyHigherZCR) {
    Real sr = 22050;

    ArrayXr y_low = generate_sine(100.0, sr, 0.5);
    ArrayXr y_high = generate_sine(1000.0, sr, 0.5);

    ArrayXXr zcr_low = zero_crossing_rate(y_low);
    ArrayXXr zcr_high = zero_crossing_rate(y_high);

    EXPECT_LT(zcr_low.mean(), zcr_high.mean());
}

// ============================================================================
// Tonnetz Tests
// ============================================================================

TEST(TonnetzTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr ton = tonnetz(y, sr);

    EXPECT_EQ(ton.rows(), 6);  // 6 tonnetz dimensions
    EXPECT_GT(ton.cols(), 0);
}

TEST(TonnetzTest, FromChroma) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr chroma = chroma_stft(y, sr);
    ArrayXXr ton = tonnetz(chroma);

    EXPECT_EQ(ton.rows(), 6);
    EXPECT_EQ(ton.cols(), chroma.cols());
}

// ============================================================================
// Polynomial Features Tests
// ============================================================================

TEST(PolyFeaturesTest, BasicComputation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    ArrayXXr poly = poly_features(y, sr);

    EXPECT_EQ(poly.rows(), 2);  // Default order=1: 2 coefficients
    EXPECT_GT(poly.cols(), 0);
}

TEST(PolyFeaturesTest, HigherOrder) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    int order = 3;
    ArrayXXr poly = poly_features(y, sr, 2048, 512, order);

    EXPECT_EQ(poly.rows(), order + 1);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(SpectralFeaturesTest, ShortSignal) {
    Real sr = 22050;
    // Very short signal
    ArrayXr y = generate_sine(440.0, sr, 0.1);

    // All features should handle short signals
    EXPECT_NO_THROW(melspectrogram(y, sr));
    EXPECT_NO_THROW(mfcc(y, sr));
    EXPECT_NO_THROW(chroma_stft(y, sr));
    EXPECT_NO_THROW(spectral_centroid(y, sr));
    EXPECT_NO_THROW(spectral_bandwidth(y, sr));
    EXPECT_NO_THROW(spectral_rolloff(y, sr));
    EXPECT_NO_THROW(spectral_flatness(y));
    EXPECT_NO_THROW(rms(y));
}

TEST(SpectralFeaturesTest, SilentSignal) {
    Real sr = 22050;
    ArrayXr y = ArrayXr::Zero(static_cast<int>(sr * 0.5));

    // Features should handle silent signals without crashing
    EXPECT_NO_THROW(melspectrogram(y, sr));
    EXPECT_NO_THROW(mfcc(y, sr));
    EXPECT_NO_THROW(spectral_centroid(y, sr));
    EXPECT_NO_THROW(rms(y));
}

// ============================================================================
// Delta Tests
// ============================================================================

TEST(DeltaTest, ConstantSignal) {
    // Delta of constant signal should be zero
    ArrayXXr data = ArrayXXr::Ones(3, 20) * 5.0;
    ArrayXXr d = delta(data, 9, 1, -1, "interp");

    EXPECT_EQ(d.rows(), 3);
    EXPECT_EQ(d.cols(), 20);
    EXPECT_LT(d.abs().maxCoeff(), 1e-10);
}

TEST(DeltaTest, LinearRamp) {
    // First derivative of linear ramp should be constant
    ArrayXXr data(1, 20);
    for (int i = 0; i < 20; ++i) {
        data(0, i) = static_cast<Real>(i);
    }

    ArrayXXr d = delta(data, 9, 1, -1, "interp");

    EXPECT_EQ(d.rows(), 1);
    EXPECT_EQ(d.cols(), 20);
    // Interior values should be close to 1.0 (slope of the ramp)
    for (int i = 4; i < 16; ++i) {
        EXPECT_NEAR(d(0, i), 1.0, 1e-10);
    }
}

TEST(DeltaTest, QuadraticOrder2) {
    // Second derivative of x^2 should be constant (2.0)
    ArrayXXr data(1, 20);
    for (int i = 0; i < 20; ++i) {
        data(0, i) = static_cast<Real>(i * i);
    }

    ArrayXXr d2 = delta(data, 5, 2, -1, "interp");

    EXPECT_EQ(d2.rows(), 1);
    EXPECT_EQ(d2.cols(), 20);
    // Interior values should be close to 2.0
    for (int i = 2; i < 18; ++i) {
        EXPECT_NEAR(d2(0, i), 2.0, 1e-8);
    }
}

TEST(DeltaTest, WidthValidation) {
    ArrayXXr data = ArrayXXr::Ones(3, 20);

    // Width < 3 should throw
    EXPECT_THROW(delta(data, 1), ParameterError);
    EXPECT_THROW(delta(data, 2), ParameterError);

    // Even width should throw
    EXPECT_THROW(delta(data, 4), ParameterError);
    EXPECT_THROW(delta(data, 8), ParameterError);

    // Order <= 0 should throw
    EXPECT_THROW(delta(data, 9, 0), ParameterError);
    EXPECT_THROW(delta(data, 9, -1), ParameterError);
}

TEST(DeltaTest, InterpModeWidthConstraint) {
    // Width > data length should throw for interp mode
    ArrayXXr data(1, 5);
    data.setOnes();

    EXPECT_THROW(delta(data, 9, 1, -1, "interp"), ParameterError);

    // But should work for other modes
    EXPECT_NO_THROW(delta(data, 9, 1, -1, "constant"));
    EXPECT_NO_THROW(delta(data, 9, 1, -1, "nearest"));
}

TEST(DeltaTest, DifferentModes) {
    ArrayXXr data(1, 20);
    for (int i = 0; i < 20; ++i) {
        data(0, i) = static_cast<Real>(i);
    }

    // All modes should produce valid output
    EXPECT_NO_THROW(delta(data, 9, 1, -1, "interp"));
    EXPECT_NO_THROW(delta(data, 9, 1, -1, "constant"));
    EXPECT_NO_THROW(delta(data, 9, 1, -1, "nearest"));
    EXPECT_NO_THROW(delta(data, 9, 1, -1, "mirror"));
    EXPECT_NO_THROW(delta(data, 9, 1, -1, "wrap"));
}

TEST(DeltaTest, Axis0) {
    // Apply along axis 0 (rows/features)
    ArrayXXr data(20, 3);
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 3; ++j) {
            data(i, j) = static_cast<Real>(i);
        }
    }

    ArrayXXr d = delta(data, 9, 1, 0, "interp");

    EXPECT_EQ(d.rows(), 20);
    EXPECT_EQ(d.cols(), 3);
    // Interior values should be close to 1.0
    for (int i = 4; i < 16; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(d(i, j), 1.0, 1e-10);
        }
    }
}

// ============================================================================
// Chroma CQT Tests
// ============================================================================

TEST(ChromaCQTTest, OutputShape) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr chroma = chroma_cqt(y, sr);

    EXPECT_EQ(chroma.rows(), 12);  // Default n_chroma
    EXPECT_GT(chroma.cols(), 0);
}

TEST(ChromaCQTTest, PureTonePeak) {
    Real sr = 22050;
    // A4 = 440 Hz, should peak at chroma bin for A
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr chroma = chroma_cqt(y, sr);

    // Check that some energy is present
    EXPECT_GT(chroma.maxCoeff(), 0);

    // Find the dominant chroma bin (averaged over time)
    ArrayXr avg_chroma = chroma.rowwise().mean();
    Eigen::Index peak_bin;
    avg_chroma.maxCoeff(&peak_bin);

    // A is chroma bin 9 in C-based chroma (C=0, C#=1, ..., A=9)
    EXPECT_EQ(peak_bin, 9);
}

TEST(ChromaCQTTest, NonNegative) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr chroma = chroma_cqt(y, sr);

    EXPECT_TRUE((chroma >= 0).all());
}

TEST(ChromaCQTTest, FromPrecomputed) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    int bins_per_octave = 36;
    int n_octaves = 7;
    int n_bins = n_octaves * bins_per_octave;
    ArrayXXc C_complex = cqt(y, sr, 512, std::nullopt, n_bins, bins_per_octave);
    ArrayXXr C = magnitude(C_complex);

    ArrayXXr chroma = chroma_cqt(C, std::nullopt,
                                  std::numeric_limits<Real>::infinity(),
                                  0.0, 12, bins_per_octave);

    EXPECT_EQ(chroma.rows(), 12);
    EXPECT_GT(chroma.cols(), 0);
    EXPECT_TRUE((chroma >= 0).all());
}

TEST(ChromaCQTTest, BinsPerOctaveValidation) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 0.5);

    // bins_per_octave must be multiple of n_chroma
    // 35 is not a multiple of 12
    EXPECT_THROW(chroma_cqt(y, sr, 512, std::nullopt,
                             std::numeric_limits<Real>::infinity(),
                             0.0, std::nullopt, 12, 7, 35),
                 ParameterError);
}

// ============================================================================
// Chroma CENS Tests
// ============================================================================

TEST(ChromaCENSTest, OutputShape) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr cens = chroma_cens(y, sr);

    EXPECT_EQ(cens.rows(), 12);
    EXPECT_GT(cens.cols(), 0);
}

TEST(ChromaCENSTest, QuantizedRange) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr cens = chroma_cens(y, sr);

    // All values should be in [0, 1] range
    EXPECT_TRUE((cens >= -1e-10).all());
    EXPECT_TRUE((cens <= 1.0 + 1e-10).all());
}

TEST(ChromaCENSTest, L2Normalized) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr cens = chroma_cens(y, sr);

    // Each column should have L2 norm approximately 1 (where there is energy)
    for (Eigen::Index t = 0; t < cens.cols(); ++t) {
        Real col_norm = std::sqrt(cens.col(t).square().sum());
        if (col_norm > 1e-6) {
            EXPECT_NEAR(col_norm, 1.0, 0.01);
        }
    }
}

TEST(ChromaCENSTest, NonNegative) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr cens = chroma_cens(y, sr);
    EXPECT_TRUE((cens >= -1e-10).all());
}

// ============================================================================
// Chroma VQT Tests
// ============================================================================

TEST(ChromaVQTTest, OutputShape) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    int bins_per_octave = 12;
    ArrayXXr chroma = chroma_vqt(y, sr, 512, std::nullopt,
                                  std::numeric_limits<Real>::infinity(),
                                  0.0, 7, bins_per_octave);

    // Output chroma bins = bins_per_octave
    EXPECT_EQ(chroma.rows(), bins_per_octave);
    EXPECT_GT(chroma.cols(), 0);
}

TEST(ChromaVQTTest, NonNegative) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    ArrayXXr chroma = chroma_vqt(y, sr);

    EXPECT_TRUE((chroma >= 0).all());
}

TEST(ChromaVQTTest, FromPrecomputed) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    int bins_per_octave = 12;
    int n_octaves = 7;
    int n_bins = n_octaves * bins_per_octave;
    ArrayXXc V_complex = vqt(y, sr, 512, std::nullopt, n_bins,
                              std::nullopt, bins_per_octave);
    ArrayXXr V = magnitude(V_complex);

    ArrayXXr chroma = chroma_vqt(V, std::nullopt,
                                  std::numeric_limits<Real>::infinity(),
                                  0.0, bins_per_octave);

    EXPECT_EQ(chroma.rows(), bins_per_octave);
    EXPECT_GT(chroma.cols(), 0);
    EXPECT_TRUE((chroma >= 0).all());
}

TEST(ChromaVQTTest, DifferentBinsPerOctave) {
    Real sr = 22050;
    ArrayXr y = generate_sine(440.0, sr, 1.0);

    int bins_per_octave = 24;
    ArrayXXr chroma = chroma_vqt(y, sr, 512, std::nullopt,
                                  std::numeric_limits<Real>::infinity(),
                                  0.0, 7, bins_per_octave);

    EXPECT_EQ(chroma.rows(), bins_per_octave);
}
