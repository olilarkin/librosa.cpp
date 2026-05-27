#include <gtest/gtest.h>
#include <librosa/core/convert.hpp>
#include <librosa/core/notation.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;

// ============================================================================
// Frame/Sample/Time Conversion Tests
// ============================================================================

class FrameToSamplesTest : public ::testing::TestWithParam<std::tuple<int, std::optional<int>>> {};

TEST_P(FrameToSamplesTest, ScalarConversion) {
    auto [hop_length, n_fft] = GetParam();

    Eigen::Index frames = 100;
    Eigen::Index samples = frames_to_samples(frames, hop_length, n_fft);

    if (!n_fft.has_value()) {
        EXPECT_EQ(samples, frames * hop_length);
    } else {
        EXPECT_EQ(samples, frames * hop_length + n_fft.value() / 2);
    }
}

TEST_P(FrameToSamplesTest, ArrayConversion) {
    auto [hop_length, n_fft] = GetParam();

    ArrayXr frames = ArrayXr::LinSpaced(10, 0, 9);
    ArrayXr samples = frames_to_samples(frames, hop_length, n_fft);

    EXPECT_EQ(samples.size(), frames.size());

    if (!n_fft.has_value()) {
        for (Eigen::Index i = 0; i < frames.size(); ++i) {
            EXPECT_NEAR(samples(i), frames(i) * hop_length, 1e-10);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(FrameToSamplesTests, FrameToSamplesTest,
    ::testing::Values(
        std::make_tuple(512, std::nullopt),
        std::make_tuple(1024, std::nullopt),
        std::make_tuple(512, 1024),
        std::make_tuple(1024, 1024)
    ));

class SamplesToFramesTest : public ::testing::TestWithParam<std::tuple<int, std::optional<int>>> {};

TEST_P(SamplesToFramesTest, ScalarConversion) {
    auto [hop_length, n_fft] = GetParam();

    Eigen::Index samples = 1024 * 100;
    Eigen::Index frames = samples_to_frames(samples, hop_length, n_fft);

    if (!n_fft.has_value()) {
        EXPECT_EQ(samples, frames * hop_length);
    }
}

INSTANTIATE_TEST_SUITE_P(SamplesToFramesTests, SamplesToFramesTest,
    ::testing::Values(
        std::make_tuple(512, std::nullopt),
        std::make_tuple(1024, std::nullopt),
        std::make_tuple(512, 1024),
        std::make_tuple(1024, 1024)
    ));

TEST(TimeConversionTest, TimeToSamples) {
    Real sr = 22050;
    ArrayXr times(3);
    times << 0, 1, 2;

    ArrayXr expected(3);
    expected << 0, sr, 2 * sr;

    ArrayXr samples = time_to_samples(times, sr);

    for (Eigen::Index i = 0; i < times.size(); ++i) {
        EXPECT_NEAR(samples(i), expected(i), 1e-5);
    }
}

TEST(TimeConversionTest, SamplesToTime) {
    Real sr = 22050;
    ArrayXr samples(3);
    samples << 0, sr, 2 * sr;

    ArrayXr times = samples_to_time(samples, sr);

    EXPECT_NEAR(times(0), 0.0, 1e-10);
    EXPECT_NEAR(times(1), 1.0, 1e-10);
    EXPECT_NEAR(times(2), 2.0, 1e-10);
}

// ============================================================================
// MIDI/Hz Conversion Tests
// ============================================================================

TEST(MidiHzTest, MidiToHz) {
    ArrayXr midi(4);
    midi << 33, 45, 57, 69;

    ArrayXr hz = midi_to_hz(midi);

    EXPECT_NEAR(hz(0), 55.0, 0.1);
    EXPECT_NEAR(hz(1), 110.0, 0.1);
    EXPECT_NEAR(hz(2), 220.0, 0.1);
    EXPECT_NEAR(hz(3), 440.0, 0.1);
}

TEST(MidiHzTest, HzToMidi) {
    EXPECT_NEAR(hz_to_midi(55.0), 33.0, 1e-5);

    ArrayXr hz(4);
    hz << 55, 110, 220, 440;

    ArrayXr midi = hz_to_midi(hz);

    EXPECT_NEAR(midi(0), 33.0, 1e-5);
    EXPECT_NEAR(midi(1), 45.0, 1e-5);
    EXPECT_NEAR(midi(2), 57.0, 1e-5);
    EXPECT_NEAR(midi(3), 69.0, 1e-5);
}

// ============================================================================
// Note Parsing Tests
// ============================================================================

class NoteToMidiTest : public ::testing::TestWithParam<std::tuple<std::string, Real>> {};

TEST_P(NoteToMidiTest, ParseNote) {
    auto [note, expected_midi] = GetParam();
    Real midi = note_to_midi(note, true);
    EXPECT_NEAR(midi, std::round(expected_midi), 1e-5);
}

INSTANTIATE_TEST_SUITE_P(NoteToMidiTests, NoteToMidiTest,
    ::testing::Values(
        std::make_tuple("C1", 24.0),
        std::make_tuple("C#1", 25.0),
        std::make_tuple("Db1", 25.0),
        std::make_tuple("C2", 36.0),
        std::make_tuple("A4", 69.0),
        std::make_tuple("A#4", 70.0),
        std::make_tuple("Bb4", 70.0)
    ));

TEST(NoteToMidiTest, BadNoteThrows) {
    EXPECT_THROW(note_to_midi("does not pass"), ParameterError);
}

TEST(NoteToHzTest, BasicNoteToHz) {
    Real hz = note_to_hz("A4");
    EXPECT_NEAR(hz, 440.0, 0.1);

    hz = note_to_hz("A5");
    EXPECT_NEAR(hz, 880.0, 0.1);

    hz = note_to_hz("A3");
    EXPECT_NEAR(hz, 220.0, 0.1);
}

TEST(NoteToHzTest, BadNoteThrows) {
    EXPECT_THROW(note_to_hz("does not pass"), ParameterError);
}

TEST(MidiToNoteTest, BasicMidiToNote) {
    std::string note = midi_to_note(69.0, true, false, false);
    EXPECT_EQ(note, "A4");

    note = midi_to_note(60.0, true, false, false);
    EXPECT_EQ(note, "C4");
}

TEST(MidiToNoteTest, CentsNoOctaveThrows) {
    EXPECT_THROW(midi_to_note(24.25, false, true), ParameterError);
}

TEST(HzToNoteTest, BasicHzToNote) {
    std::string note = hz_to_note(440.0, true, false, false);
    EXPECT_EQ(note, "A4");

    note = hz_to_note(880.0, true, false, false);
    EXPECT_EQ(note, "A5");
}

// ============================================================================
// Mel Scale Tests
// ============================================================================

TEST(MelScaleTest, HzToMel) {
    // Test Slaney formula (default)
    Real mel = hz_to_mel(1000.0, false);
    EXPECT_GT(mel, 0);

    // Test HTK formula
    Real mel_htk = hz_to_mel(1000.0, true);
    EXPECT_GT(mel_htk, 0);
}

TEST(MelScaleTest, MelToHz) {
    // Roundtrip test
    Real hz = 1000.0;
    Real mel = hz_to_mel(hz, false);
    Real hz_back = mel_to_hz(mel, false);
    EXPECT_NEAR(hz_back, hz, 1e-5);

    // HTK roundtrip
    mel = hz_to_mel(hz, true);
    hz_back = mel_to_hz(mel, true);
    EXPECT_NEAR(hz_back, hz, 1e-5);
}

TEST(MelScaleTest, ArrayConversion) {
    ArrayXr hz(4);
    hz << 100, 500, 1000, 5000;

    ArrayXr mel = hz_to_mel(hz, false);
    ArrayXr hz_back = mel_to_hz(mel, false);

    for (Eigen::Index i = 0; i < hz.size(); ++i) {
        EXPECT_NEAR(hz_back(i), hz(i), 1e-3);
    }
}

// ============================================================================
// Octave Conversion Tests
// ============================================================================

class OctsToHzTest : public ::testing::TestWithParam<std::tuple<Real, int>> {};

TEST_P(OctsToHzTest, BasicConversion) {
    auto [tuning, bins_per_octave] = GetParam();

    ArrayXr octs(4);
    octs << 1, 2, 3, 4;

    ArrayXr expected(4);
    expected << 55, 110, 220, 440;
    expected *= std::pow(2.0, tuning / bins_per_octave);

    ArrayXr hz = octs_to_hz(octs, tuning, bins_per_octave);

    for (Eigen::Index i = 0; i < octs.size(); ++i) {
        EXPECT_NEAR(hz(i), expected(i), 1e-3);
    }
}

INSTANTIATE_TEST_SUITE_P(OctsToHzTests, OctsToHzTest,
    ::testing::Values(
        std::make_tuple(0.0, 12),
        std::make_tuple(-0.2, 12),
        std::make_tuple(0.1, 12),
        std::make_tuple(0.0, 24),
        std::make_tuple(0.0, 36)
    ));

class HzToOctsTest : public ::testing::TestWithParam<std::tuple<Real, int>> {};

TEST_P(HzToOctsTest, BasicConversion) {
    auto [tuning, bins_per_octave] = GetParam();

    ArrayXr expected(4);
    expected << 1, 2, 3, 4;

    ArrayXr freq(4);
    freq << 55, 110, 220, 440;
    freq *= std::pow(2.0, tuning / bins_per_octave);

    ArrayXr octs = hz_to_octs(freq, tuning, bins_per_octave);

    for (Eigen::Index i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(octs(i), expected(i), 1e-5);
    }
}

INSTANTIATE_TEST_SUITE_P(HzToOctsTests, HzToOctsTest,
    ::testing::Values(
        std::make_tuple(0.0, 12),
        std::make_tuple(-0.2, 12),
        std::make_tuple(0.1, 12),
        std::make_tuple(0.0, 24),
        std::make_tuple(0.0, 36)
    ));

// ============================================================================
// A4/Tuning Conversion Tests
// ============================================================================

TEST(A4TuningTest, A4ToTuning) {
    // Standard A4 = 440 Hz should give tuning = 0
    Real tuning = A4_to_tuning(440.0, 12);
    EXPECT_NEAR(tuning, 0.0, 1e-5);

    // A4 = 432 Hz should give negative tuning
    tuning = A4_to_tuning(432.0, 12);
    EXPECT_LT(tuning, 0);
}

TEST(A4TuningTest, TuningToA4) {
    // Zero tuning should give 440 Hz
    Real A4 = tuning_to_A4(0.0, 12);
    EXPECT_NEAR(A4, 440.0, 1e-5);
}

TEST(A4TuningTest, Roundtrip) {
    Real A4_orig = 444.0;
    Real tuning = A4_to_tuning(A4_orig, 24);
    Real A4_back = tuning_to_A4(tuning, 24);
    EXPECT_NEAR(A4_back, A4_orig, 1e-5);
}

// ============================================================================
// FFT Frequencies Tests
// ============================================================================

class FFTFrequenciesTest : public ::testing::TestWithParam<std::tuple<Real, int>> {};

TEST_P(FFTFrequenciesTest, BasicFFTFrequencies) {
    auto [sr, n_fft] = GetParam();

    ArrayXr freqs = fft_frequencies(sr, n_fft);

    // DC component should be 0
    EXPECT_NEAR(freqs(0), 0.0, 1e-10);

    // Nyquist should be sr/2
    EXPECT_NEAR(freqs(freqs.size() - 1), sr / 2.0, 1e-5);

    // Frequencies should increase linearly
    ArrayXr diffs = freqs.tail(freqs.size() - 1) - freqs.head(freqs.size() - 1);
    Real expected_spacing = sr / n_fft;
    for (Eigen::Index i = 0; i < diffs.size(); ++i) {
        EXPECT_NEAR(diffs(i), expected_spacing, 1e-5);
    }
}

INSTANTIATE_TEST_SUITE_P(FFTFrequenciesTests, FFTFrequenciesTest,
    ::testing::Values(
        std::make_tuple(8000, 1024),
        std::make_tuple(22050, 1024),
        std::make_tuple(22050, 2048),
        std::make_tuple(44100, 2048)
    ));

// ============================================================================
// CQT Frequencies Tests
// ============================================================================

class CQTFrequenciesTest : public ::testing::TestWithParam<std::tuple<int, Real, int, Real>> {};

TEST_P(CQTFrequenciesTest, BasicCQTFrequencies) {
    auto [n_bins, fmin, bins_per_octave, tuning] = GetParam();

    ArrayXr freqs = cqt_frequencies(n_bins, fmin, bins_per_octave, tuning);

    // Check correct number of bins
    EXPECT_EQ(freqs.size(), n_bins);

    // First bin should match fmin adjusted by tuning
    Real expected_fmin = fmin * std::pow(2.0, tuning / bins_per_octave);
    EXPECT_NEAR(freqs(0), expected_fmin, 1e-3);

    // Check constant Q (log spacing)
    if (n_bins > 1) {
        ArrayXr log_freqs = freqs.log() / std::log(2.0);
        ArrayXr Q = log_freqs.tail(n_bins - 1) - log_freqs.head(n_bins - 1);

        Real expected_Q = 1.0 / bins_per_octave;
        for (Eigen::Index i = 0; i < Q.size(); ++i) {
            EXPECT_NEAR(Q(i), expected_Q, 1e-5);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(CQTFrequenciesTests, CQTFrequenciesTest,
    ::testing::Values(
        std::make_tuple(12, 440.0, 12, 0.0),
        std::make_tuple(24, 440.0, 12, 0.0),
        std::make_tuple(36, 440.0, 24, 0.0),
        std::make_tuple(12, 440.0, 12, -0.25),
        std::make_tuple(12, 440.0, 12, 0.25)
    ));

// ============================================================================
// Mel Frequencies Tests
// ============================================================================

TEST(MelFrequenciesTest, BasicMelFrequencies) {
    ArrayXr freqs = mel_frequencies(128, 0.0, 11025.0, false);

    EXPECT_EQ(freqs.size(), 128);
    EXPECT_NEAR(freqs(0), 0.0, 1e-10);
    EXPECT_NEAR(freqs(freqs.size() - 1), 11025.0, 1e-3);
}

// ============================================================================
// Tempo Frequencies Tests
// ============================================================================

class TempoFrequenciesTest : public ::testing::TestWithParam<std::tuple<int, int, Real>> {};

TEST_P(TempoFrequenciesTest, BasicTempoFrequencies) {
    auto [n_bins, hop_length, sr] = GetParam();

    ArrayXr freqs = tempo_frequencies(n_bins, hop_length, sr);

    // Verify length
    EXPECT_EQ(freqs.size(), n_bins);

    // 0-bin should be infinite
    EXPECT_FALSE(std::isfinite(freqs(0)));

    // Remaining bins should be positive and spaced correctly
    if (n_bins > 1) {
        // Convert to periods and check spacing
        ArrayXr invfreqs = 60.0 * sr / freqs.tail(n_bins - 1);
        EXPECT_NEAR(invfreqs(0), hop_length, 1e-5);

        ArrayXr diffs = invfreqs.tail(invfreqs.size() - 1) - invfreqs.head(invfreqs.size() - 1);
        for (Eigen::Index i = 0; i < diffs.size(); ++i) {
            EXPECT_NEAR(diffs(i), hop_length, 1e-5);
        }
    }
}

TEST(TempoFrequenciesTest, InvalidNBinsThrows) {
    EXPECT_THROW(tempo_frequencies(0, 512, 22050), ParameterError);
}

INSTANTIATE_TEST_SUITE_P(TempoFrequenciesTests, TempoFrequenciesTest,
    ::testing::Values(
        std::make_tuple(1, 512, 22050),
        std::make_tuple(16, 512, 22050),
        std::make_tuple(128, 256, 11025)
    ));

// ============================================================================
// Frequency Weighting Tests
// ============================================================================

TEST(AWeightingTest, At1kHz) {
    Real a_khz = A_weighting(1000.0, std::nullopt);
    // A-weighting should be approximately 0 dB at 1 kHz
    EXPECT_NEAR(a_khz, 0.0, 1e-2);
}

TEST(AWeightingTest, MinDbCap) {
    Real min_db = -40.0;
    ArrayXr freqs = ArrayXr::LinSpaced(100, 20.0, 20000.0);
    ArrayXr a_range = A_weighting(freqs, min_db);

    // Check that the db cap works
    for (Eigen::Index i = 0; i < a_range.size(); ++i) {
        EXPECT_GE(a_range(i), min_db);
    }
}

TEST(BWeightingTest, At1kHz) {
    Real b_khz = B_weighting(1000.0, std::nullopt);
    EXPECT_NEAR(b_khz, 0.0, 1e-2);
}

TEST(CWeightingTest, At1kHz) {
    Real c_khz = C_weighting(1000.0, std::nullopt);
    EXPECT_NEAR(c_khz, 0.0, 1e-2);
}

TEST(DWeightingTest, At1kHz) {
    Real d_khz = D_weighting(1000.0, std::nullopt);
    EXPECT_NEAR(d_khz, 0.0, 1e-2);
}

TEST(ZWeightingTest, AllZero) {
    ArrayXr freqs = ArrayXr::LinSpaced(100, 20.0, 20000.0);
    ArrayXr z_range = Z_weighting(freqs, std::nullopt);

    // Z-weighting should always be 0
    for (Eigen::Index i = 0; i < z_range.size(); ++i) {
        EXPECT_NEAR(z_range(i), 0.0, 1e-10);
    }
}

// ============================================================================
// Block Conversion Tests
// ============================================================================

class BlocksToFramesTest : public ::testing::TestWithParam<int> {};

TEST_P(BlocksToFramesTest, BasicConversion) {
    int block_length = GetParam();

    Eigen::Index blocks = 10;
    Eigen::Index frames = blocks_to_frames(blocks, block_length);

    EXPECT_EQ(frames, blocks * block_length);
}

TEST_P(BlocksToFramesTest, ArrayConversion) {
    int block_length = GetParam();

    ArrayXr blocks(2);
    blocks << 10, 20;

    ArrayXr frames = blocks_to_frames(blocks, block_length);

    EXPECT_NEAR(frames(0), 10 * block_length, 1e-10);
    EXPECT_NEAR(frames(1), 20 * block_length, 1e-10);
}

INSTANTIATE_TEST_SUITE_P(BlocksToFramesTests, BlocksToFramesTest,
    ::testing::Values(1, 4, 8));

class BlocksToSamplesTest : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(BlocksToSamplesTest, BasicConversion) {
    auto [block_length, hop_length] = GetParam();

    Eigen::Index blocks = 10;
    Eigen::Index samples = blocks_to_samples(blocks, block_length, hop_length);

    EXPECT_EQ(samples, blocks * block_length * hop_length);
}

INSTANTIATE_TEST_SUITE_P(BlocksToSamplesTests, BlocksToSamplesTest,
    ::testing::Values(
        std::make_tuple(1, 512),
        std::make_tuple(4, 512),
        std::make_tuple(8, 256)
    ));

class BlocksToTimeTest : public ::testing::TestWithParam<std::tuple<int, int, Real>> {};

TEST_P(BlocksToTimeTest, BasicConversion) {
    auto [block_length, hop_length, sr] = GetParam();

    Eigen::Index blocks = 10;
    Real time = blocks_to_time(blocks, block_length, hop_length, sr);

    Real expected = static_cast<Real>(blocks * block_length * hop_length) / sr;
    EXPECT_NEAR(time, expected, 1e-10);
}

INSTANTIATE_TEST_SUITE_P(BlocksToTimeTests, BlocksToTimeTest,
    ::testing::Values(
        std::make_tuple(1, 512, 22050),
        std::make_tuple(4, 512, 22050),
        std::make_tuple(8, 256, 44100)
    ));

// ============================================================================
// Samples/Times Like Tests
// ============================================================================

TEST(SamplesLikeTest, BasicSamplesLike) {
    ArrayXXr X = ArrayXXr::Ones(3, 5);
    int hop_length = 512;

    ArrayXr samples = samples_like(X, hop_length, std::nullopt, -1);

    // Should have 5 samples (one per column)
    EXPECT_EQ(samples.size(), 5);

    for (Eigen::Index i = 0; i < samples.size(); ++i) {
        EXPECT_NEAR(samples(i), i * hop_length, 1e-10);
    }
}

TEST(TimesLikeTest, BasicTimesLike) {
    ArrayXXr X = ArrayXXr::Ones(3, 5);
    Real sr = 22050;
    int hop_length = 512;

    ArrayXr times = times_like(X, sr, hop_length, std::nullopt, -1);

    EXPECT_EQ(times.size(), 5);

    for (Eigen::Index i = 0; i < times.size(); ++i) {
        Real expected = static_cast<Real>(i * hop_length) / sr;
        EXPECT_NEAR(times(i), expected, 1e-10);
    }
}

// ============================================================================
// Svara Conversion Tests
// ============================================================================

TEST(SvaraTest, MidiToSvaraH_Sa) {
    // C4 = MIDI 60, with Sa at 60 => should be "S" (Sa)
    EXPECT_EQ(midi_to_svara_h(60, 60), "S");
}

TEST(SvaraTest, MidiToSvaraH_Pa) {
    // Perfect fifth up from Sa: MIDI 67, Sa=60 => Pa
    EXPECT_EQ(midi_to_svara_h(67, 60), "P");
}

TEST(SvaraTest, MidiToSvaraH_LongForm) {
    EXPECT_EQ(midi_to_svara_h(60, 60, false), "Sa");
    EXPECT_EQ(midi_to_svara_h(62, 60, false), "Re");
    EXPECT_EQ(midi_to_svara_h(67, 60, false), "Pa");
}

TEST(SvaraTest, MidiToSvaraH_UpperOctave) {
    // MIDI 72 = Sa one octave up (svara_num=12)
    auto svara = midi_to_svara_h(72, 60, true, true, false);
    EXPECT_EQ(svara, "S'");
}

TEST(SvaraTest, MidiToSvaraH_LowerOctave) {
    // MIDI 48 = Sa one octave down (svara_num=-12)
    auto svara = midi_to_svara_h(48, 60, true, true, false);
    EXPECT_EQ(svara, "S,");
}

TEST(SvaraTest, MidiToSvaraC_Kanakangi) {
    // Mela #1 (kanakangi): uses R1, G1, D1, N1
    EXPECT_EQ(midi_to_svara_c(60, 60, 1), "S");
    EXPECT_EQ(midi_to_svara_c(67, 60, 1), "P");
}

TEST(SvaraTest, MidiToSvaraC_ByName) {
    auto s1 = midi_to_svara_c(60, 60, "kanakangi");
    auto s2 = midi_to_svara_c(60, 60, 1);
    EXPECT_EQ(s1, s2);
}

TEST(SvaraTest, HzToSvaraH) {
    // 440 Hz = A4, with Sa at 261.63 (C4)
    // A is 9 semitones above C => "D" (Dha) in Hindustani
    auto svara = hz_to_svara_h(440.0, 261.63);
    EXPECT_EQ(svara, "D");
}

TEST(SvaraTest, NoteToSvaraH) {
    auto svara = note_to_svara_h("C4", "C4");
    EXPECT_EQ(svara, "S");
    auto svara2 = note_to_svara_h("G4", "C4");
    EXPECT_EQ(svara2, "P");
}

TEST(SvaraTest, NoteToSvaraC) {
    auto svara = note_to_svara_c("C4", "C4", 1);
    EXPECT_EQ(svara, "S");
}

TEST(SvaraTest, HzToFjs_Unison) {
    auto note = hz_to_fjs(440.0, 440.0, "A", false);
    EXPECT_EQ(note, "A");
}

TEST(SvaraTest, HzToFjs_PerfectFifth) {
    auto note = hz_to_fjs(660.0, 440.0, "A", false);
    // 660/440 = 3/2 = perfect fifth above A = E
    EXPECT_EQ(note, "E");
}

// ============================================================================
// FJS Notation Tests
// ============================================================================

TEST(FjsTest, PerfectFifth) {
    auto note = interval_to_fjs(3.0/2.0, "C", 65.0/63.0, false);
    EXPECT_EQ(note, "G");
}

TEST(FjsTest, PerfectFourth) {
    auto note = interval_to_fjs(4.0/3.0, "C", 65.0/63.0, false);
    EXPECT_EQ(note, "F");
}

TEST(FjsTest, MajorThirdWith5) {
    // 5/4 = Ptolemaic major third, should be "E^5" (ASCII) or "E⁵" (unicode)
    auto note = interval_to_fjs(5.0/4.0, "C", 65.0/63.0, false);
    EXPECT_EQ(note, "E^5");
}

TEST(FjsTest, MinorThirdWith5) {
    // 6/5 = minor third, should have subscript 5
    auto note = interval_to_fjs(6.0/5.0, "C", 65.0/63.0, false);
    EXPECT_EQ(note, "Eb_5");
}

TEST(FjsTest, Unison) {
    auto note = interval_to_fjs(1.0, "C", 65.0/63.0, false);
    EXPECT_EQ(note, "C");
}

TEST(FjsTest, Octave) {
    auto note = interval_to_fjs(2.0, "C", 65.0/63.0, false);
    EXPECT_EQ(note, "C");
}

TEST(FjsTest, InvalidInterval) {
    EXPECT_THROW(interval_to_fjs(0.0), ParameterError);
    EXPECT_THROW(interval_to_fjs(-1.0), ParameterError);
}
