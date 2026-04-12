#pragma once

#include "../types.hpp"
#include <optional>
#include <limits>

namespace librosa {

// ============================================================================
// Pitch Tuning
// ============================================================================

/// Estimate tuning offset from a collection of pitch frequencies
/// @param frequencies Array of pitch frequencies in Hz
/// @param resolution Resolution of tuning as fraction of a bin (0.01 = cents)
/// @param bins_per_octave Number of frequency bins per octave
/// @return Estimated tuning deviation in fractions of a bin [-0.5, 0.5)
Real pitch_tuning(const ArrayXr& frequencies,
                  Real resolution = 0.01,
                  int bins_per_octave = 12);

/// Estimate tuning from audio or spectrogram
/// @param y Audio time series (optional)
/// @param sr Sample rate
/// @param S Magnitude or power spectrogram (optional)
/// @param n_fft FFT size
/// @param resolution Tuning resolution
/// @param bins_per_octave Bins per octave
/// @param fmin Minimum frequency for pitch detection
/// @param fmax Maximum frequency for pitch detection
/// @param threshold Magnitude threshold for pitch detection
/// @return Estimated tuning deviation
Real estimate_tuning(const ArrayXr& y,
                     Real sr = 22050,
                     int n_fft = 2048,
                     Real resolution = 0.01,
                     int bins_per_octave = 12,
                     Real fmin = 150.0,
                     Real fmax = 4000.0,
                     Real threshold = 0.1);

Real estimate_tuning(const ArrayXXr& S,
                     Real sr = 22050,
                     int n_fft = 2048,
                     Real resolution = 0.01,
                     int bins_per_octave = 12,
                     Real fmin = 150.0,
                     Real fmax = 4000.0,
                     Real threshold = 0.1);

// ============================================================================
// Pitch Tracking
// ============================================================================

/// Pitch tracking using parabolic interpolation on STFT
/// @param y Audio time series (optional, provide y or S)
/// @param sr Sample rate
/// @param n_fft FFT size
/// @param hop_length Hop size in samples
/// @param fmin Minimum frequency
/// @param fmax Maximum frequency
/// @param threshold Magnitude threshold (relative to reference)
/// @param center Whether to center frames
/// @return Pair of (pitches, magnitudes) arrays
std::pair<ArrayXXr, ArrayXXr> piptrack(
    const ArrayXr& y,
    Real sr = 22050,
    int n_fft = 2048,
    std::optional<int> hop_length = std::nullopt,
    Real fmin = 150.0,
    Real fmax = 4000.0,
    Real threshold = 0.1,
    bool center = true);

std::pair<ArrayXXr, ArrayXXr> piptrack(
    const ArrayXXr& S,
    Real sr = 22050,
    int n_fft = 2048,
    std::optional<int> hop_length = std::nullopt,
    Real fmin = 150.0,
    Real fmax = 4000.0,
    Real threshold = 0.1);

// ============================================================================
// YIN Algorithm
// ============================================================================

/// Fundamental frequency estimation using the YIN algorithm
/// @param y Audio time series
/// @param fmin Minimum frequency in Hz (recommended: C2 ~= 65 Hz)
/// @param fmax Maximum frequency in Hz (recommended: C7 ~= 2093 Hz)
/// @param sr Sample rate
/// @param frame_length Frame length in samples
/// @param hop_length Hop length (default: frame_length / 4)
/// @param trough_threshold Absolute threshold for peak estimation
/// @param center Whether to center frames
/// @return Time series of fundamental frequencies in Hz
ArrayXr yin(const ArrayXr& y,
            Real fmin,
            Real fmax,
            Real sr = 22050,
            int frame_length = 2048,
            std::optional<int> hop_length = std::nullopt,
            Real trough_threshold = 0.1,
            bool center = true,
            PadMode pad_mode = PadMode::Constant);

// ============================================================================
// pYIN Algorithm
// ============================================================================

/// Result type for pyin()
struct PyinResult {
    ArrayXr f0;           ///< Fundamental frequencies (Hz), NaN for unvoiced
    Eigen::Array<bool, Eigen::Dynamic, 1> voiced_flag;  ///< Voiced/unvoiced per frame
    ArrayXr voiced_prob;  ///< Probability of voicing per frame
};

/// Fundamental frequency estimation using probabilistic YIN (pYIN)
/// @param y Audio time series
/// @param fmin Minimum frequency in Hz
/// @param fmax Maximum frequency in Hz
/// @param sr Sample rate
/// @param frame_length Frame length in samples
/// @param hop_length Hop length (default: frame_length / 4)
/// @param n_thresholds Number of thresholds for peak estimation
/// @param beta_a Shape parameter a for Beta distribution prior
/// @param beta_b Shape parameter b for Beta distribution prior
/// @param boltzmann_parameter Shape parameter for Boltzmann distribution prior
/// @param resolution Pitch bin resolution (0.01 = cents)
/// @param max_transition_rate Maximum pitch transition rate in octaves/second
/// @param switch_prob Probability of switching voiced/unvoiced
/// @param no_trough_prob Probability mass for no-trough case
/// @param fill_na Value for unvoiced frames (NaN by default; nullopt = best guess)
/// @param center Whether to center frames
/// @param pad_mode Padding mode
/// @return PyinResult with f0, voiced_flag, voiced_prob arrays
PyinResult pyin(
    const ArrayXr& y,
    Real fmin,
    Real fmax,
    Real sr = 22050,
    int frame_length = 2048,
    std::optional<int> hop_length = std::nullopt,
    int n_thresholds = 100,
    Real beta_a = 2.0,
    Real beta_b = 18.0,
    Real boltzmann_parameter = 2.0,
    Real resolution = 0.1,
    Real max_transition_rate = 35.92,
    Real switch_prob = 0.01,
    Real no_trough_prob = 0.01,
    std::optional<Real> fill_na = std::numeric_limits<Real>::quiet_NaN(),
    bool center = true,
    PadMode pad_mode = PadMode::Constant);

} // namespace librosa
