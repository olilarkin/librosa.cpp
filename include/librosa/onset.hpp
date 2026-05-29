#pragma once

#include "types.hpp"
#include <optional>
#include <vector>

namespace librosa {
namespace onset {

// ============================================================================
// Utility Functions
// ============================================================================

/// Apply a 1D maximum filter along an axis
/// @param S Input array
/// @param size Filter size
/// @param axis Axis along which to filter (0=rows, -2=freq for spectrograms)
/// @return Filtered array
ArrayXXr maximum_filter1d(const ArrayXXr& S, int size, int axis = -2);

/// Match events from one set to another (find nearest matches)
/// @param events_from Source events to match
/// @param events_to Target events to match against
/// @param left If true, allow matching to events to the left
/// @param right If true, allow matching to events to the right
/// @return Indices into events_to for each event in events_from
std::vector<Eigen::Index> match_events(
    const std::vector<Eigen::Index>& events_from,
    const std::vector<Eigen::Index>& events_to,
    bool left = true,
    bool right = true);

// ============================================================================
// Onset Strength Functions
// ============================================================================

/// Compute spectral flux onset strength envelope
/// @param y Audio time series (optional if S is provided)
/// @param sr Sample rate
/// @param S Pre-computed spectrogram (optional)
/// @param n_fft FFT size for mel spectrogram
/// @param hop_length Hop length
/// @param lag Time lag for computing differences
/// @param max_size Size of max filter for vibrato suppression (1 = disabled)
/// @param detrend Remove DC component if true
/// @param center Center the onset envelope
/// @return Onset strength envelope
ArrayXr onset_strength(
    const ArrayXr& y,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    int lag = 1,
    int max_size = 1,
    bool detrend = false,
    bool center = true,
    AggregateFunc aggregate = AggregateFunc::Mean);

/// Compute onset strength from pre-computed spectrogram
ArrayXr onset_strength(
    const ArrayXXr& S,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    int lag = 1,
    int max_size = 1,
    bool detrend = false,
    bool center = true,
    AggregateFunc aggregate = AggregateFunc::Mean);

/// Compute spectral flux onset strength envelope across multiple channels
/// @param y Audio time series (optional if S is provided)
/// @param sr Sample rate
/// @param n_fft FFT size for mel spectrogram
/// @param hop_length Hop length
/// @param lag Time lag for computing differences
/// @param max_size Size of max filter for vibrato suppression (1 = disabled)
/// @param detrend Remove DC component if true
/// @param center Center the onset envelope
/// @param channels Channel boundary indices (e.g., {0, 32, 64, 96, 128})
///        If empty, a single channel spanning all bands is used.
/// @return Onset strength envelope [shape: (n_channels, n_frames)]
ArrayXXr onset_strength_multi(
    const ArrayXr& y,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    int lag = 1,
    int max_size = 1,
    bool detrend = false,
    bool center = true,
    const std::vector<int>& channels = {});

/// Compute onset strength multi from pre-computed spectrogram
ArrayXXr onset_strength_multi(
    const ArrayXXr& S,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    int lag = 1,
    int max_size = 1,
    bool detrend = false,
    bool center = true,
    const std::vector<int>& channels = {});

// ============================================================================
// Onset Detection
// ============================================================================

/// Units for onset detection output
enum class OnsetUnits {
    Frames,
    Samples,
    Time
};

/// Locate note onset events by picking peaks in an onset strength envelope
/// @param y Audio time series
/// @param sr Sample rate
/// @param hop_length Hop length for onset envelope
/// @param backtrack If true, backtrack onsets to nearest minimum
/// @param units Output units. Use onset_detect_times for real-valued time output.
/// @param normalize Normalize onset envelope before detection
/// @param pre_max Samples before n for max computation (0 = use default)
/// @param post_max Samples after n for max computation (0 = use default)
/// @param pre_avg Samples before n for mean computation (0 = use default)
/// @param post_avg Samples after n for mean computation (0 = use default)
/// @param delta Threshold offset for mean
/// @param wait Samples to wait after picking a peak (0 = use default)
/// @return Vector of onset positions
std::vector<Eigen::Index> onset_detect(
    const ArrayXr& y,
    Real sr = 22050,
    int hop_length = 512,
    bool backtrack = false,
    OnsetUnits units = OnsetUnits::Frames,
    bool normalize = true,
    int pre_max = 0,
    int post_max = 0,
    int pre_avg = 0,
    int post_avg = 0,
    Real delta = 0.07,
    int wait = 0);

/// Detect onsets using pre-computed onset envelope
std::vector<Eigen::Index> onset_detect_envelope(
    const ArrayXr& onset_envelope,
    Real sr = 22050,
    int hop_length = 512,
    bool backtrack = false,
    OnsetUnits units = OnsetUnits::Frames,
    bool normalize = true,
    int pre_max = 0,
    int post_max = 0,
    int pre_avg = 0,
    int post_avg = 0,
    Real delta = 0.07,
    int wait = 0);

/// Detect onsets and return times (convenience function)
ArrayXr onset_detect_times(
    const ArrayXr& y,
    Real sr = 22050,
    int hop_length = 512,
    bool backtrack = false,
    bool normalize = true);

// ============================================================================
// Onset Backtracking
// ============================================================================

/// Backtrack detected onset events to the nearest preceding local minimum
/// @param events Onset event frame indices
/// @param energy Energy function (e.g., onset envelope or RMS)
/// @return Backtracked onset positions
std::vector<Eigen::Index> onset_backtrack(
    const std::vector<Eigen::Index>& events,
    const ArrayXr& energy);

} // namespace onset
} // namespace librosa
