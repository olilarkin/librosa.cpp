#pragma once

#include "types.hpp"
#include <optional>
#include <vector>
#include <utility>

namespace librosa {
namespace beat {

// ============================================================================
// Tempogram
// ============================================================================

/// Compute the tempogram: local autocorrelation of the onset strength envelope
/// @param onset_envelope Pre-computed onset strength envelope
/// @param sr Sample rate
/// @param hop_length Hop length
/// @param win_length Length of the autocorrelation window (in frames)
/// @param center If true, center the autocorrelation windows
/// @param window Window type
/// @param norm Normalization mode (inf, -inf, 0, or p-norm; nullopt to disable)
/// @return Tempogram [shape=(win_length, n_frames)]
ArrayXXr tempogram(
    const ArrayXr& onset_envelope,
    Real sr = 22050,
    int hop_length = 512,
    int win_length = 384,
    bool center = true,
    WindowType window = WindowType::Hann,
    std::optional<Real> norm = std::numeric_limits<Real>::infinity());

/// Compute tempogram from audio
ArrayXXr tempogram_audio(
    const ArrayXr& y,
    Real sr,
    int hop_length = 512,
    int win_length = 384,
    bool center = true,
    WindowType window = WindowType::Hann,
    std::optional<Real> norm = std::numeric_limits<Real>::infinity());

// ============================================================================
// Tempo Estimation
// ============================================================================

/// Estimate the tempo (beats per minute)
/// @param onset_envelope Pre-computed onset strength envelope
/// @param sr Sample rate
/// @param hop_length Hop length
/// @param start_bpm Initial guess for tempo
/// @param std_bpm Standard deviation of tempo distribution
/// @param ac_size Length (in seconds) of the auto-correlation window
/// @param max_tempo Maximum tempo to consider (nullopt for no limit)
/// @param aggregate If true, return global tempo; if false, per-frame tempo
/// @return Estimated tempo in BPM
Real tempo(
    const ArrayXr& onset_envelope,
    Real sr = 22050,
    int hop_length = 512,
    Real start_bpm = 120.0,
    Real std_bpm = 1.0,
    Real ac_size = 8.0,
    std::optional<Real> max_tempo = 320.0,
    bool aggregate = true);

/// Estimate tempo from audio
Real tempo_audio(
    const ArrayXr& y,
    Real sr,
    int hop_length = 512,
    Real start_bpm = 120.0,
    Real std_bpm = 1.0,
    Real ac_size = 8.0,
    std::optional<Real> max_tempo = 320.0,
    bool aggregate = true);

/// Estimate per-frame tempo (dynamic tempo)
ArrayXr tempo_frames(
    const ArrayXr& onset_envelope,
    Real sr = 22050,
    int hop_length = 512,
    Real start_bpm = 120.0,
    Real std_bpm = 1.0,
    Real ac_size = 8.0,
    std::optional<Real> max_tempo = 320.0);

// ============================================================================
// Beat Tracking
// ============================================================================

/// Units for beat tracking output
enum class BeatUnits {
    Frames,
    Samples,
    Time
};

/// Dynamic programming beat tracker
/// @param onset_envelope Pre-computed onset strength envelope
/// @param sr Sample rate
/// @param hop_length Hop length
/// @param start_bpm Initial tempo guess
/// @param tightness How closely to adhere to tempo estimate
/// @param trim Trim leading/trailing weak beats
/// @param bpm Optional fixed tempo (if not provided, estimate from signal)
/// @param units Output units. Use beat_track_times for real-valued time output.
/// @return Pair of (tempo, beat positions)
std::pair<Real, std::vector<Eigen::Index>> beat_track(
    const ArrayXr& onset_envelope,
    Real sr = 22050,
    int hop_length = 512,
    Real start_bpm = 120.0,
    Real tightness = 100.0,
    bool trim = true,
    std::optional<Real> bpm = std::nullopt,
    BeatUnits units = BeatUnits::Frames);

/// Beat track from audio
std::pair<Real, std::vector<Eigen::Index>> beat_track_audio(
    const ArrayXr& y,
    Real sr,
    int hop_length = 512,
    Real start_bpm = 120.0,
    Real tightness = 100.0,
    bool trim = true,
    std::optional<Real> bpm = std::nullopt,
    BeatUnits units = BeatUnits::Frames);

/// Convenience function: beat track and return times
std::pair<Real, ArrayXr> beat_track_times(
    const ArrayXr& y,
    Real sr,
    int hop_length = 512,
    Real start_bpm = 120.0,
    Real tightness = 100.0,
    bool trim = true);

// ============================================================================
// Predominant Local Pulse (PLP)
// ============================================================================

/// Compute predominant local pulse (PLP) from an onset strength envelope
///
/// Uses the Fourier tempogram to find the dominant pulse in each frame,
/// then synthesizes a pulse train at the local tempo.
///
/// @param onset_envelope Pre-computed onset strength envelope
/// @param sr Sample rate
/// @param hop_length Hop length
/// @param win_length Length of the Fourier tempogram window (in frames)
/// @param tempo_min Minimum tempo in BPM (nullopt to disable)
/// @param tempo_max Maximum tempo in BPM (nullopt to disable)
/// @return Predominant local pulse [shape: (n_frames,)]
ArrayXr plp(const ArrayXr& onset_envelope, Real sr = 22050,
            int hop_length = 512, int win_length = 384,
            std::optional<Real> tempo_min = 30,
            std::optional<Real> tempo_max = 300);

/// Compute PLP from audio
ArrayXr plp_audio(const ArrayXr& y, Real sr = 22050,
                  int hop_length = 512, int win_length = 384,
                  std::optional<Real> tempo_min = 30,
                  std::optional<Real> tempo_max = 300);

} // namespace beat
} // namespace librosa
