#include <gtest/gtest.h>
#include <librosa/filters.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;
using namespace librosa::filters;

// ============================================================================
// Mel Filter Bank Tests
// ============================================================================

TEST(MelFilterTest, OutputShape) {
    int sr = 22050;
    int n_fft = 2048;
    int n_mels = 128;

    ArrayXXr melfb = mel(sr, n_fft, n_mels);

    EXPECT_EQ(melfb.rows(), n_mels);
    EXPECT_EQ(melfb.cols(), 1 + n_fft / 2);
}

class MelFilterParamsTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(MelFilterParamsTest, ValidOutput) {
    auto [n_fft, n_mels, sr] = GetParam();

    ArrayXXr melfb = mel(sr, n_fft, n_mels);

    // All values should be non-negative
    EXPECT_TRUE((melfb >= 0).all());

    // Each mel band should have some non-zero values
    for (int i = 0; i < n_mels; ++i) {
        Real row_max = melfb.row(i).maxCoeff();
        EXPECT_GT(row_max, 0.0);
    }
}

INSTANTIATE_TEST_SUITE_P(MelFilterParamsTests, MelFilterParamsTest,
    ::testing::Values(
        std::make_tuple(512, 40, 16000),
        std::make_tuple(2048, 128, 22050),
        std::make_tuple(4096, 80, 44100)
    ));

TEST(MelFilterTest, FrequencyRanges) {
    int sr = 22050;
    int n_fft = 2048;

    // With custom fmax
    ArrayXXr melfb1 = mel(sr, n_fft, 128, 0.0, 8000.0);
    ArrayXXr melfb2 = mel(sr, n_fft, 128, 0.0, std::nullopt);

    // Default fmax = sr/2 should give different results
    // (more energy in higher bins)
    EXPECT_FALSE((melfb1 - melfb2).abs().maxCoeff() < 1e-10);
}

TEST(MelFilterTest, NormalizationModes) {
    int sr = 22050;
    int n_fft = 2048;
    int n_mels = 40;

    ArrayXXr mel_none = mel(sr, n_fft, n_mels, 0.0, std::nullopt, false, MelNorm::None);
    ArrayXXr mel_slaney = mel(sr, n_fft, n_mels, 0.0, std::nullopt, false, MelNorm::Slaney);
    ArrayXXr mel_l1 = mel(sr, n_fft, n_mels, 0.0, std::nullopt, false, MelNorm::L1);
    ArrayXXr mel_l2 = mel(sr, n_fft, n_mels, 0.0, std::nullopt, false, MelNorm::L2);

    // L1 normalized rows should sum to 1
    for (int i = 0; i < n_mels; ++i) {
        EXPECT_NEAR(mel_l1.row(i).sum(), 1.0, 1e-5);
    }

    // L2 normalized rows should have unit norm
    for (int i = 0; i < n_mels; ++i) {
        Real norm = std::sqrt((mel_l2.row(i) * mel_l2.row(i)).sum());
        EXPECT_NEAR(norm, 1.0, 1e-5);
    }

    // None (no normalization): peak values should be <= 1 and > 0
    for (int i = 0; i < n_mels; ++i) {
        EXPECT_GT(mel_none.row(i).maxCoeff(), 0.0);
        EXPECT_LE(mel_none.row(i).maxCoeff(), 1.0 + 1e-10);
    }
}

TEST(MelFilterTest, HTKFormula) {
    int sr = 22050;
    int n_fft = 2048;

    ArrayXXr mel_slaney = mel(sr, n_fft, 40, 0.0, std::nullopt, false);
    ArrayXXr mel_htk = mel(sr, n_fft, 40, 0.0, std::nullopt, true);

    // HTK and Slaney should produce different filter banks
    EXPECT_GT((mel_slaney - mel_htk).abs().maxCoeff(), 1e-5);
}

// ============================================================================
// Chroma Filter Bank Tests
// ============================================================================

TEST(ChromaFilterTest, OutputShape) {
    int sr = 22050;
    int n_fft = 4096;
    int n_chroma = 12;

    ArrayXXr chromafb = chroma(sr, n_fft, n_chroma);

    EXPECT_EQ(chromafb.rows(), n_chroma);
    EXPECT_EQ(chromafb.cols(), 1 + n_fft / 2);
}

class ChromaFilterParamsTest : public ::testing::TestWithParam<int> {};

TEST_P(ChromaFilterParamsTest, DifferentChromaBins) {
    int n_chroma = GetParam();
    int sr = 22050;
    int n_fft = 4096;

    ArrayXXr chromafb = chroma(sr, n_fft, n_chroma);

    EXPECT_EQ(chromafb.rows(), n_chroma);

    // All values should be non-negative
    EXPECT_TRUE((chromafb >= 0).all());
}

INSTANTIATE_TEST_SUITE_P(ChromaFilterParamsTests, ChromaFilterParamsTest,
    ::testing::Values(12, 24, 36));

TEST(ChromaFilterTest, TuningEffect) {
    int sr = 22050;
    int n_fft = 4096;

    ArrayXXr chroma_tuned = chroma(sr, n_fft, 12, 0.5);
    ArrayXXr chroma_untuned = chroma(sr, n_fft, 12, 0.0);

    // Tuning should shift the filter bank
    EXPECT_GT((chroma_tuned - chroma_untuned).abs().maxCoeff(), 1e-5);
}

TEST(ChromaFilterTest, OctwidthEffect) {
    int sr = 22050;
    int n_fft = 4096;

    ArrayXXr chroma_narrow = chroma(sr, n_fft, 12, 0.0, 5.0, 1.0);
    ArrayXXr chroma_wide = chroma(sr, n_fft, 12, 0.0, 5.0, 4.0);
    ArrayXXr chroma_flat = chroma(sr, n_fft, 12, 0.0, 5.0, std::nullopt);

    // Different octave widths should produce different results
    EXPECT_GT((chroma_narrow - chroma_wide).abs().maxCoeff(), 1e-5);
    EXPECT_GT((chroma_flat - chroma_narrow).abs().maxCoeff(), 1e-5);
}

// ============================================================================
// Window Bandwidth Tests
// ============================================================================

TEST(WindowBandwidthTest, KnownValues) {
    // Test against known values from WINDOW_BANDWIDTHS
    EXPECT_NEAR(window_bandwidth("hann"), 1.50018310546875, 1e-5);
    EXPECT_NEAR(window_bandwidth("hamming"), 1.3629455320350348, 1e-5);
    EXPECT_NEAR(window_bandwidth("blackman"), 1.7269681554262326, 1e-5);
    EXPECT_NEAR(window_bandwidth("rectangular"), 1.0, 1e-5);
}

TEST(WindowBandwidthTest, WindowTypeEnum) {
    EXPECT_NEAR(window_bandwidth(WindowType::Hann), 1.50018310546875, 1e-5);
    EXPECT_NEAR(window_bandwidth(WindowType::Hamming), 1.3629455320350348, 1e-5);
    EXPECT_NEAR(window_bandwidth(WindowType::Rectangular), 1.0, 1e-5);
}

// ============================================================================
// CQ to Chroma Tests
// ============================================================================

TEST(CQToChromaTest, OutputShape) {
    int n_input = 84;  // 7 octaves
    int bins_per_octave = 12;
    int n_chroma = 12;

    ArrayXXr cq_chr = cq_to_chroma(n_input, bins_per_octave, n_chroma);

    EXPECT_EQ(cq_chr.rows(), n_chroma);
    EXPECT_EQ(cq_chr.cols(), n_input);
}

TEST(CQToChromaTest, SumsToOne) {
    // Each CQ bin should contribute to exactly one chroma bin
    int n_input = 84;
    ArrayXXr cq_chr = cq_to_chroma(n_input);

    for (int j = 0; j < n_input; ++j) {
        Real col_sum = cq_chr.col(j).sum();
        EXPECT_NEAR(col_sum, 1.0, 1e-5);
    }
}

TEST(CQToChromaTest, IncompatibleMergeThrows) {
    // 12 bins per octave, but 5 chroma bins is incompatible
    EXPECT_THROW(cq_to_chroma(84, 12, 5), ParameterError);
}

class CQToChromaParamsTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(CQToChromaParamsTest, ValidOutput) {
    auto [n_input, bins_per_octave, n_chroma] = GetParam();

    ArrayXXr cq_chr = cq_to_chroma(n_input, bins_per_octave, n_chroma);

    EXPECT_EQ(cq_chr.rows(), n_chroma);
    EXPECT_EQ(cq_chr.cols(), n_input);

    // All values should be non-negative
    EXPECT_TRUE((cq_chr >= 0).all());
}

INSTANTIATE_TEST_SUITE_P(CQToChromaParamsTests, CQToChromaParamsTest,
    ::testing::Values(
        std::make_tuple(84, 12, 12),
        std::make_tuple(168, 24, 12),
        std::make_tuple(48, 12, 12)
    ));

// ============================================================================
// Wavelet Length Tests
// ============================================================================

TEST(WaveletLengthsTest, BasicOutput) {
    ArrayXr freqs = cqt_frequencies(84, note_to_hz("C1"), 12);

    auto [lengths, f_cutoff] = wavelet_lengths(freqs, 22050);

    EXPECT_EQ(lengths.size(), 84);
    EXPECT_TRUE((lengths > 0).all());
    EXPECT_GT(f_cutoff, 0);
}

TEST(WaveletLengthsTest, FilterScaleEffect) {
    ArrayXr freqs = cqt_frequencies(24, note_to_hz("C4"), 12);

    auto [lengths1, _1] = wavelet_lengths(freqs, 22050, WindowType::Hann, 1.0);
    auto [lengths2, _2] = wavelet_lengths(freqs, 22050, WindowType::Hann, 2.0);

    // Larger filter scale should produce longer filters
    for (Eigen::Index i = 0; i < lengths1.size(); ++i) {
        EXPECT_GT(lengths2(i), lengths1(i));
    }
}

TEST(WaveletLengthsTest, GammaEffect) {
    ArrayXr freqs = cqt_frequencies(24, note_to_hz("C4"), 12);

    auto [lengths_cq, _1] = wavelet_lengths(freqs, 22050, WindowType::Hann, 1.0, 0.0);
    auto [lengths_vq, _2] = wavelet_lengths(freqs, 22050, WindowType::Hann, 1.0, 24.0);

    // VQT (gamma > 0) should produce shorter filters at low frequencies
    EXPECT_LT(lengths_vq(0), lengths_cq(0));
}

TEST(WaveletLengthsTest, InvalidFreqsThrow) {
    ArrayXr neg_freqs(3);
    neg_freqs << -100, 200, 400;
    EXPECT_THROW(wavelet_lengths(neg_freqs, 22050), ParameterError);

    ArrayXr unsorted(3);
    unsorted << 400, 200, 300;
    EXPECT_THROW(wavelet_lengths(unsorted, 22050), ParameterError);
}

TEST(WaveletLengthsTest, InvalidParamsThrow) {
    ArrayXr freqs(3);
    freqs << 100, 200, 400;

    EXPECT_THROW(wavelet_lengths(freqs, 22050, WindowType::Hann, -1.0), ParameterError);
    EXPECT_THROW(wavelet_lengths(freqs, 22050, WindowType::Hann, 1.0, -1.0), ParameterError);
}

// ============================================================================
// Wavelet Basis Tests
// ============================================================================

TEST(WaveletTest, OutputShape) {
    ArrayXr freqs = cqt_frequencies(24, note_to_hz("C4"), 12);

    auto [filters, lengths] = wavelet(freqs, 22050);

    EXPECT_EQ(filters.rows(), 24);
    EXPECT_EQ(lengths.size(), 24);

    // Padded length should be power of 2
    int n = static_cast<int>(filters.cols());
    EXPECT_EQ(n & (n - 1), 0);  // Power of 2 check
}

TEST(WaveletTest, NoPadding) {
    ArrayXr freqs = cqt_frequencies(12, note_to_hz("C4"), 12);

    auto [filters_padded, _1] = wavelet(freqs, 22050, WindowType::Hann, 1.0, true);
    auto [filters_unpadded, _2] = wavelet(freqs, 22050, WindowType::Hann, 1.0, false);

    // Padded should have power-of-2 length
    int n_padded = static_cast<int>(filters_padded.cols());
    EXPECT_EQ(n_padded & (n_padded - 1), 0);

    // Unpadded may not
    EXPECT_LE(filters_unpadded.cols(), filters_padded.cols());
}

// ============================================================================
// Diagonal Filter Tests
// ============================================================================

TEST(DiagonalFilterTest, OutputShape) {
    int n = 21;

    ArrayXXr filt = diagonal_filter(WindowType::Hann, n);

    EXPECT_EQ(filt.rows(), n);
    EXPECT_EQ(filt.cols(), n);
}

TEST(DiagonalFilterTest, SumsToOne) {
    ArrayXXr filt = diagonal_filter(WindowType::Hann, 21);

    EXPECT_NEAR(filt.sum(), 1.0, 1e-5);
}

TEST(DiagonalFilterTest, ZeroMean) {
    ArrayXXr filt = diagonal_filter(WindowType::Hann, 21, 1.0, std::nullopt, true);

    EXPECT_NEAR(filt.mean(), 0.0, 1e-10);
}

TEST(DiagonalFilterTest, NonNegativeDefault) {
    ArrayXXr filt = diagonal_filter(WindowType::Hann, 21, 1.0, std::nullopt, false);

    EXPECT_TRUE((filt >= 0).all());
}

// ============================================================================
// Multi-rate Frequencies Tests
// ============================================================================

TEST(MRFrequenciesTest, OutputLength) {
    auto [freqs, srs] = mr_frequencies(0.0);

    EXPECT_EQ(freqs.size(), 85);  // C0 to B7 = 85 notes
    EXPECT_EQ(srs.size(), 85);
}

TEST(MRFrequenciesTest, TuningEffect) {
    auto [freqs_tuned, _1] = mr_frequencies(0.5);
    auto [freqs_untuned, _2] = mr_frequencies(0.0);

    // Tuning should shift frequencies
    EXPECT_GT((freqs_tuned - freqs_untuned).abs().maxCoeff(), 0.1);
}

TEST(MRFrequenciesTest, SampleRateRanges) {
    auto [freqs, srs] = mr_frequencies(0.0);

    // Check sample rate regions
    for (int i = 0; i < 36; ++i) {
        EXPECT_NEAR(srs(i), 882.0, 1e-5);
    }
    for (int i = 36; i < 70; ++i) {
        EXPECT_NEAR(srs(i), 4410.0, 1e-5);
    }
    for (int i = 70; i < 85; ++i) {
        EXPECT_NEAR(srs(i), 22050.0, 1e-5);
    }
}
