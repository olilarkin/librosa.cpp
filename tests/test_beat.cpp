#include <gtest/gtest.h>
#include <librosa/beat.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;
using namespace librosa::beat;

// ============================================================================
// Tempogram Tests
// ============================================================================

TEST(TempogramTest, BasicTempogram) {
    // Create a synthetic onset envelope with periodicity
    int n_frames = 200;
    ArrayXr onset_envelope(n_frames);

    // Add periodic peaks
    for (int i = 0; i < n_frames; ++i) {
        // Period of ~20 frames (roughly 120 BPM at sr=22050, hop=512)
        onset_envelope(i) = 0.5 + 0.5 * std::cos(2 * M_PI * i / 20.0);
    }

    ArrayXXr tg = tempogram(onset_envelope, 22050, 512, 100);

    EXPECT_EQ(tg.rows(), 100);  // win_length
    EXPECT_GT(tg.cols(), 0);
}

TEST(TempogramTest, WithCentering) {
    ArrayXr onset_envelope(100);
    onset_envelope.setRandom();
    onset_envelope = onset_envelope.abs();

    ArrayXXr tg_centered = tempogram(onset_envelope, 22050, 512, 50, true);
    ArrayXXr tg_not_centered = tempogram(onset_envelope, 22050, 512, 50, false);

    EXPECT_EQ(tg_centered.rows(), 50);
    EXPECT_EQ(tg_not_centered.rows(), 50);

    // Centered should have same number of frames as input
    EXPECT_EQ(tg_centered.cols(), onset_envelope.size());
}

TEST(TempogramTest, InvalidWinLength) {
    ArrayXr onset_envelope(100);
    onset_envelope.setRandom();

    EXPECT_THROW(tempogram(onset_envelope, 22050, 512, 0), ParameterError);
    EXPECT_THROW(tempogram(onset_envelope, 22050, 512, -1), ParameterError);
}

// ============================================================================
// Tempo Estimation Tests
// ============================================================================

TEST(TempoTest, BasicTempo) {
    // Create onset envelope with clear periodicity
    int n_frames = 500;
    ArrayXr onset_envelope(n_frames);

    // Simulate 120 BPM at sr=22050, hop=512
    // frames per beat = 22050/512 * 60/120 = ~21.5 frames
    int period = 22;
    for (int i = 0; i < n_frames; ++i) {
        onset_envelope(i) = (i % period == 0) ? 1.0 : 0.1;
    }

    Real estimated_tempo = tempo(onset_envelope, 22050, 512, 120.0);

    // 120 BPM lands between frame lags for sr=22050/hop=512, so the tempo
    // estimate quantizes to the nearest autocorrelation lag.
    Real expected_tempo = 60.0 * 22050.0 / (512.0 * period);
    EXPECT_NEAR(estimated_tempo, expected_tempo, 1e-9);
    EXPECT_NEAR(estimated_tempo, 120.0, 3.0);
}

TEST(TempoTest, Exact120BPMFrameGrid) {
    int n_frames = 24 * 32;
    ArrayXr onset_envelope(n_frames);

    int period = 24;
    for (int i = 0; i < n_frames; ++i) {
        onset_envelope(i) = (i % period == 0) ? 1.0 : 0.0;
    }

    Real estimated_tempo = tempo(onset_envelope, 24000, 500, 120.0);

    EXPECT_NEAR(estimated_tempo, 120.0, 1e-9);
}

TEST(TempoTest, SlowTempo) {
    int n_frames = 500;
    ArrayXr onset_envelope(n_frames);

    // Simulate 60 BPM - period of ~43 frames
    int period = 43;
    for (int i = 0; i < n_frames; ++i) {
        onset_envelope(i) = (i % period == 0) ? 1.0 : 0.1;
    }

    Real estimated_tempo = tempo(onset_envelope, 22050, 512, 60.0);

    EXPECT_GT(estimated_tempo, 0);
}

TEST(TempoTest, InvalidStartBpm) {
    ArrayXr onset_envelope(100);
    onset_envelope.setRandom();

    EXPECT_THROW(tempo(onset_envelope, 22050, 512, 0.0), ParameterError);
    EXPECT_THROW(tempo(onset_envelope, 22050, 512, -10.0), ParameterError);
}

TEST(TempoTest, MaxTempoConstraint) {
    int n_frames = 500;
    ArrayXr onset_envelope(n_frames);

    // Fast periodicity
    int period = 10;  // ~260 BPM
    for (int i = 0; i < n_frames; ++i) {
        onset_envelope(i) = (i % period == 0) ? 1.0 : 0.1;
    }

    Real tempo_unconstrained = tempo(onset_envelope, 22050, 512, 120.0, 1.0, 8.0, std::nullopt);
    Real tempo_constrained = tempo(onset_envelope, 22050, 512, 120.0, 1.0, 8.0, 200.0);

    // Constrained tempo should be <= max_tempo
    EXPECT_LE(tempo_constrained, 200.0);
}

// ============================================================================
// Beat Tracking Tests
// ============================================================================

TEST(BeatTrackTest, BasicBeatTrack) {
    // Create synthetic onset envelope
    int n_frames = 300;
    ArrayXr onset_envelope(n_frames);

    // Periodic beats
    int period = 22;  // ~120 BPM
    for (int i = 0; i < n_frames; ++i) {
        onset_envelope(i) = (i % period == 0) ? 1.0 : 0.1;
    }

    auto [bpm, beats] = beat_track(onset_envelope, 22050, 512);

    EXPECT_GT(bpm, 0);
    EXPECT_GT(beats.size(), 0);

    // Beats should be roughly periodic
    if (beats.size() > 1) {
        for (size_t i = 1; i < beats.size(); ++i) {
            Eigen::Index diff = beats[i] - beats[i - 1];
            // Should be roughly period frames apart (with some tolerance)
            EXPECT_GT(diff, period / 2);
            EXPECT_LT(diff, period * 2);
        }
    }
}

TEST(BeatTrackTest, WithFixedTempo) {
    int n_frames = 300;
    ArrayXr onset_envelope(n_frames);

    int period = 22;
    for (int i = 0; i < n_frames; ++i) {
        onset_envelope(i) = (i % period == 0) ? 1.0 : 0.1;
    }

    auto [bpm, beats] = beat_track(onset_envelope, 22050, 512, 120.0, 100.0, true, 120.0);

    EXPECT_EQ(bpm, 120.0);
    EXPECT_GT(beats.size(), 0);
}

TEST(BeatTrackTest, EmptySignal) {
    ArrayXr onset_envelope = ArrayXr::Zero(100);

    auto [bpm, beats] = beat_track(onset_envelope, 22050, 512);

    EXPECT_EQ(bpm, 0.0);
    EXPECT_EQ(beats.size(), 0);
}

TEST(BeatTrackTest, UnitsConversion) {
    int n_frames = 300;
    ArrayXr onset_envelope(n_frames);

    int period = 22;
    for (int i = 0; i < n_frames; ++i) {
        onset_envelope(i) = (i % period == 0) ? 1.0 : 0.1;
    }

    int hop_length = 512;

    auto [bpm_frames, beats_frames] = beat_track(onset_envelope, 22050, hop_length,
                                                  120.0, 100.0, true, std::nullopt, BeatUnits::Frames);
    auto [bpm_samples, beats_samples] = beat_track(onset_envelope, 22050, hop_length,
                                                    120.0, 100.0, true, std::nullopt, BeatUnits::Samples);

    EXPECT_EQ(beats_frames.size(), beats_samples.size());

    // Samples should be frames * hop_length
    for (size_t i = 0; i < beats_frames.size(); ++i) {
        EXPECT_EQ(beats_samples[i], beats_frames[i] * hop_length);
    }
}

TEST(BeatTrackTest, TimeUnitsRequireHelper) {
    ArrayXr onset_envelope = ArrayXr::Zero(100);
    onset_envelope(10) = 1.0;

    EXPECT_THROW(
        beat_track(onset_envelope, 22050, 512, 120.0, 100.0, true,
                   std::nullopt, BeatUnits::Time),
        ParameterError);
}

TEST(BeatTrackTest, NoTrimming) {
    int n_frames = 300;
    ArrayXr onset_envelope(n_frames);

    // Add some weak beats at the start and end
    int period = 22;
    for (int i = 0; i < n_frames; ++i) {
        if (i % period == 0) {
            if (i < 50 || i > 250) {
                onset_envelope(i) = 0.1;  // Weak beats
            } else {
                onset_envelope(i) = 1.0;  // Strong beats
            }
        } else {
            onset_envelope(i) = 0.05;
        }
    }

    auto [bpm_trim, beats_trim] = beat_track(onset_envelope, 22050, 512, 120.0, 100.0, true);
    auto [bpm_notrim, beats_notrim] = beat_track(onset_envelope, 22050, 512, 120.0, 100.0, false);

    // Without trimming might have more beats
    EXPECT_GE(beats_notrim.size(), beats_trim.size());
}

// ============================================================================
// Beat Track Times Test
// ============================================================================

TEST(BeatTrackTimesTest, Basic) {
    int n_frames = 300;
    ArrayXr onset_envelope(n_frames);

    int period = 22;
    for (int i = 0; i < n_frames; ++i) {
        onset_envelope(i) = (i % period == 0) ? 1.0 : 0.1;
    }

    // Need audio for beat_track_times
    // Create simple test signal
    int sr = 22050;
    int hop_length = 512;
    int n_samples = n_frames * hop_length;
    ArrayXr y = ArrayXr::Zero(n_samples);

    // Add clicks at beat positions
    for (int i = 0; i < n_frames; ++i) {
        if (i % period == 0) {
            int sample_pos = i * hop_length;
            if (sample_pos < n_samples) {
                y(sample_pos) = 1.0;
            }
        }
    }

    auto [bpm, times] = beat_track_times(y, sr, hop_length);

    EXPECT_GT(bpm, 0);
    // Times should be in seconds
    if (times.size() > 0) {
        EXPECT_GE(times(0), 0.0);
        EXPECT_LT(times(times.size() - 1), static_cast<Real>(n_samples) / sr + 1.0);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(BeatTest, ShortSignal) {
    ArrayXr onset_envelope(10);  // Very short
    onset_envelope.setConstant(0.5);

    auto [bpm, beats] = beat_track(onset_envelope, 22050, 512);

    // Should handle gracefully
    EXPECT_GE(bpm, 0);
}

TEST(BeatTest, ConstantSignal) {
    ArrayXr onset_envelope = ArrayXr::Constant(200, 0.5);

    auto [bpm, beats] = beat_track(onset_envelope, 22050, 512);

    // Constant signal has no clear beats
    // Should still return something reasonable
    EXPECT_GE(bpm, 0);
}

TEST(BeatTest, VaryingTempo) {
    // Signal that changes tempo midway
    int n_frames = 400;
    ArrayXr onset_envelope(n_frames);

    for (int i = 0; i < n_frames; ++i) {
        int period = (i < 200) ? 22 : 30;  // 120 BPM then ~90 BPM
        onset_envelope(i) = ((i < 200 ? i : i - 200) % period == 0) ? 1.0 : 0.1;
    }

    auto [bpm, beats] = beat_track(onset_envelope, 22050, 512);

    // Should still find beats
    EXPECT_GT(bpm, 0);
    EXPECT_GT(beats.size(), 0);
}
