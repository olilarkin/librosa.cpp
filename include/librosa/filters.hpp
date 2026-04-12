#pragma once

#include "types.hpp"
#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace librosa {
namespace filters {

// ============================================================================
// Window Bandwidth Constants
// ============================================================================

/// Get the equivalent noise bandwidth (ENBW) for common window functions
extern const std::unordered_map<std::string, Real> WINDOW_BANDWIDTHS;

// ============================================================================
// Mel Filter Bank
// ============================================================================

/// Mel filter normalization modes
enum class MelNorm {
    None,    ///< No normalization (peak value of 1.0)
    Slaney,  ///< Slaney-style normalization (area = constant)
    L1,      ///< L1 normalization
    L2,      ///< L2 normalization
    Inf      ///< L-infinity normalization
};

/// Create a Mel filter bank
/// @param sr Sample rate of the incoming signal
/// @param n_fft Number of FFT components
/// @param n_mels Number of Mel bands to generate
/// @param fmin Lowest frequency (Hz)
/// @param fmax Highest frequency (Hz), defaults to sr/2
/// @param htk Use HTK formula instead of Slaney
/// @param norm Normalization mode
/// @return Mel filter bank matrix (n_mels, 1 + n_fft/2)
ArrayXXr mel(Real sr, int n_fft, int n_mels = 128,
             Real fmin = 0.0, std::optional<Real> fmax = std::nullopt,
             bool htk = false, MelNorm norm = MelNorm::Slaney);

// ============================================================================
// Chroma Filter Bank
// ============================================================================

/// Create a chroma filter bank
/// @param sr Audio sampling rate
/// @param n_fft Number of FFT bins
/// @param n_chroma Number of chroma bins (pitch classes)
/// @param tuning Tuning deviation from A440 in fractions of a chroma bin
/// @param ctroct Center octave for Gaussian weighting
/// @param octwidth Octave width for Gaussian weighting (nullopt for flat)
/// @param norm Normalization factor (nullopt for none, infinity for max-norm)
/// @param base_c If true, start at C; if false, start at A
/// @return Chroma filter matrix (n_chroma, 1 + n_fft/2)
ArrayXXr chroma(Real sr, int n_fft, int n_chroma = 12,
                Real tuning = 0.0, Real ctroct = 5.0,
                std::optional<Real> octwidth = 2.0,
                std::optional<Real> norm = 2.0, bool base_c = true);

// ============================================================================
// Window Bandwidth
// ============================================================================

/// Get the equivalent noise bandwidth of a window function
/// @param window Window type
/// @param n Number of coefficients for estimation
/// @return Bandwidth in FFT bins
Real window_bandwidth(WindowType window, int n = 1000);

/// Get the equivalent noise bandwidth of a window function by name
/// @param window Window name
/// @param n Number of coefficients for estimation
/// @return Bandwidth in FFT bins
Real window_bandwidth(const std::string& window, int n = 1000);

// ============================================================================
// CQ to Chroma Transformation
// ============================================================================

/// Construct transformation matrix from CQ bins to chroma bins
/// @param n_input Number of CQ input bins
/// @param bins_per_octave Bins per octave in CQ
/// @param n_chroma Number of output chroma bins
/// @param fmin Center frequency of first CQ channel (nullopt for C1)
/// @param base_c If true, start at C; if false, start at A
/// @return Transformation matrix (n_chroma, n_input)
ArrayXXr cq_to_chroma(int n_input, int bins_per_octave = 12,
                       int n_chroma = 12,
                       std::optional<Real> fmin = std::nullopt,
                       bool base_c = true);

// ============================================================================
// Relative Bandwidth
// ============================================================================

/// Compute the relative bandwidth for each of a set of specified frequencies
/// @param freqs Center frequencies (Hz), at least 2 values
/// @return Relative bandwidth array
ArrayXr relative_bandwidth(const ArrayXr& freqs);

// ============================================================================
// Wavelet Filters
// ============================================================================

/// Return length of each filter in a wavelet basis
/// @param freqs Center frequencies (Hz), must be ascending
/// @param sr Audio sampling rate
/// @param window Window type
/// @param filter_scale Filter scale factor
/// @param gamma Bandwidth offset for VQT (0 for CQT)
/// @param alpha Optional pre-computed relative bandwidth (scalar)
/// @return Pair of (filter lengths, cutoff frequency)
std::pair<ArrayXr, Real> wavelet_lengths(
    const ArrayXr& freqs,
    Real sr = 22050,
    WindowType window = WindowType::Hann,
    Real filter_scale = 1.0,
    Real gamma = 0.0,
    std::optional<Real> alpha = std::nullopt);

/// Return length of each filter in a wavelet basis (array alpha, optional gamma)
/// @param freqs Center frequencies (Hz), must be ascending
/// @param sr Audio sampling rate
/// @param window Window type
/// @param filter_scale Filter scale factor
/// @param gamma Bandwidth offset (nullopt = auto-compute from ERB)
/// @param alpha Pre-computed relative bandwidth array (per-frequency)
/// @return Pair of (filter lengths, cutoff frequency)
std::pair<ArrayXr, Real> wavelet_lengths(
    const ArrayXr& freqs,
    Real sr,
    WindowType window,
    Real filter_scale,
    std::optional<Real> gamma,
    const ArrayXr& alpha);

/// Construct a wavelet basis using windowed complex sinusoids
/// @param freqs Center frequencies (Hz), must be ascending
/// @param sr Audio sampling rate
/// @param window Window type
/// @param filter_scale Filter scale factor
/// @param pad_fft Pad to nearest power of 2
/// @param norm Normalization type (nullopt for none)
/// @param gamma Bandwidth offset for VQT
/// @param alpha Optional pre-computed relative bandwidth (scalar)
/// @return Pair of (filters, lengths) - filters shape (n_bins, max_len)
std::pair<ArrayXXc, ArrayXr> wavelet(
    const ArrayXr& freqs,
    Real sr = 22050,
    WindowType window = WindowType::Hann,
    Real filter_scale = 1.0,
    bool pad_fft = true,
    std::optional<Real> norm = 1.0,
    Real gamma = 0.0,
    std::optional<Real> alpha = std::nullopt);

/// Construct a wavelet basis (array alpha, optional gamma)
/// @param freqs Center frequencies (Hz), must be ascending
/// @param sr Audio sampling rate
/// @param window Window type
/// @param filter_scale Filter scale factor
/// @param pad_fft Pad to nearest power of 2
/// @param norm Normalization type (nullopt for none)
/// @param gamma Bandwidth offset (nullopt = auto-compute from ERB)
/// @param alpha Pre-computed relative bandwidth array (per-frequency)
/// @return Pair of (filters, lengths) - filters shape (n_bins, max_len)
std::pair<ArrayXXc, ArrayXr> wavelet(
    const ArrayXr& freqs,
    Real sr,
    WindowType window,
    Real filter_scale,
    bool pad_fft,
    std::optional<Real> norm,
    std::optional<Real> gamma,
    const ArrayXr& alpha);

// ============================================================================
// Diagonal Filter
// ============================================================================

/// Build a 2D diagonal filter for smoothing recurrence matrices
/// @param window Window type
/// @param n Filter length
/// @param slope Diagonal slope
/// @param angle Optional angle in radians (overrides slope)
/// @param zero_mean If true, create zero-mean filter
/// @return 2D filter kernel
ArrayXXr diagonal_filter(WindowType window, int n,
                          Real slope = 1.0,
                          std::optional<Real> angle = std::nullopt,
                          bool zero_mean = false);

// ============================================================================
// Multi-rate Frequencies
// ============================================================================

/// Generate center frequencies and sample rates for pitch filterbank
/// @param tuning Tuning deviation from A440
/// @return Pair of (center_freqs, sample_rates)
std::pair<ArrayXr, ArrayXr> mr_frequencies(Real tuning = 0.0);

// ============================================================================
// IIR Filter Types
// ============================================================================

/// Second-order section coefficients: [b0, b1, b2, a0, a1, a2]
using SOSSection = std::array<Real, 6>;

/// A complete SOS filter is a cascade of second-order sections
using SOSFilter = std::vector<SOSSection>;

// ============================================================================
// Semitone Filterbank
// ============================================================================

/// Construct a multirate IIR bandpass filterbank.
/// Designs elliptic (Cauer) bandpass filters at specified center frequencies,
/// matching Python librosa's default behavior.
/// @param center_freqs Center frequencies of filters (nullopt = piano frequencies)
/// @param sample_rates Sample rate for each filter (nullopt = default from mr_frequencies)
/// @param Q Q factor (influences filter bandwidth)
/// @param passband_ripple Maximum passband loss in dB
/// @param stopband_attenuation Minimum stopband attenuation in dB
/// @return Pair of (vector of SOS filters, sample rates array)
std::pair<std::vector<SOSFilter>, ArrayXr> semitone_filterbank(
    std::optional<ArrayXr> center_freqs = std::nullopt,
    Real tuning = 0.0,
    std::optional<ArrayXr> sample_rates = std::nullopt,
    Real Q = 25.0,
    Real passband_ripple = 1.0,
    Real stopband_attenuation = 50.0);

// ============================================================================
// SOS Filtering
// ============================================================================

/// Forward SOS filtering (Direct Form II Transposed)
/// @param sos SOS filter sections
/// @param x Input signal
/// @return Filtered signal
ArrayXr sosfilt(const SOSFilter& sos, const ArrayXr& x);

/// Forward SOS filtering with initial conditions
/// @param sos SOS filter sections
/// @param x Input signal
/// @param zi Initial conditions, shape (n_sections, 2)
/// @return Pair of (filtered signal, final filter state)
std::pair<ArrayXr, ArrayXXr> sosfilt(const SOSFilter& sos, const ArrayXr& x,
                                     const ArrayXXr& zi);

/// Compute steady-state initial conditions for SOS filter
/// @param sos SOS filter sections
/// @return Initial conditions, shape (n_sections, 2)
ArrayXXr sosfilt_zi(const SOSFilter& sos);

/// Zero-phase (forward-backward) SOS filtering
/// @param sos SOS filter sections
/// @param x Input signal
/// @return Filtered signal
ArrayXr sosfiltfilt(const SOSFilter& sos, const ArrayXr& x);

} // namespace filters
} // namespace librosa
