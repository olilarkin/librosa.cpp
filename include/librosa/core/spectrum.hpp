#pragma once

#include "../types.hpp"
#include "convert.hpp"
#include <optional>
#include <functional>

namespace librosa {

// ============================================================================
// Short-Time Fourier Transform
// ============================================================================

/// Short-time Fourier transform (STFT)
/// @param y Input signal
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param win_length Window length (default: n_fft)
/// @param window Window type
/// @param center If true, pad signal so frames are centered
/// @param pad_mode Padding mode if center is true
/// @return Complex STFT matrix (n_fft/2+1, n_frames)
ArrayXXc stft(const ArrayXr& y,
              int n_fft = 2048,
              std::optional<int> hop_length = std::nullopt,
              std::optional<int> win_length = std::nullopt,
              WindowType window = WindowType::Hann,
              bool center = true,
              PadMode pad_mode = PadMode::Constant);

/// Short-time Fourier transform with pre-computed window array
/// @param y Input signal
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param window Pre-computed window array of length n_fft
/// @param center If true, pad signal so frames are centered
/// @param pad_mode Padding mode if center is true
/// @return Complex STFT matrix (n_fft/2+1, n_frames)
ArrayXXc stft(const ArrayXr& y,
              int n_fft,
              int hop_length,
              const ArrayXr& window,
              bool center = true,
              PadMode pad_mode = PadMode::Constant);

/// Inverse STFT
/// @param stft_matrix Complex STFT matrix
/// @param hop_length Samples between frames
/// @param win_length Window length
/// @param n_fft FFT size (inferred if nullopt)
/// @param window Window type
/// @param center Whether original STFT was centered
/// @param length Desired output length (nullopt for automatic)
/// @return Reconstructed time-domain signal
ArrayXr istft(const ArrayXXc& stft_matrix,
              std::optional<int> hop_length = std::nullopt,
              std::optional<int> win_length = std::nullopt,
              std::optional<int> n_fft = std::nullopt,
              WindowType window = WindowType::Hann,
              bool center = true,
              std::optional<int> length = std::nullopt);

// ============================================================================
// Magnitude and Phase
// ============================================================================

/// Separate complex spectrogram into magnitude and phase
/// @param D Complex spectrogram
/// @param power Exponent for magnitude (1 for amplitude, 2 for power)
/// @return Pair of (magnitude, phase) where phase is complex unit phasors
std::pair<ArrayXXr, ArrayXXc> magphase(const ArrayXXc& D, Real power = 1.0);

/// Compute magnitude from complex spectrogram
/// @param D Complex spectrogram
/// @return Magnitude spectrogram
ArrayXXr magnitude(const ArrayXXc& D);

/// Compute phase from complex spectrogram
/// @param D Complex spectrogram
/// @return Phase in radians
ArrayXXr phase(const ArrayXXc& D);

// ============================================================================
// Decibel Conversions
// ============================================================================

/// Convert power spectrogram to decibels
/// @param S Power spectrogram
/// @param ref Reference power (scalar or function)
/// @param amin Minimum threshold
/// @param top_db Maximum dynamic range
/// @return Decibel-scaled spectrogram
ArrayXXr power_to_db(const ArrayXXr& S,
                     Real ref = 1.0,
                     Real amin = 1e-10,
                     std::optional<Real> top_db = 80.0);

ArrayXXr power_to_db(const ArrayXXr& S,
                     std::function<Real(const ArrayXXr&)> ref,
                     Real amin = 1e-10,
                     std::optional<Real> top_db = 80.0);

Real power_to_db(Real S, Real ref = 1.0, Real amin = 1e-10,
                 std::optional<Real> top_db = std::nullopt);

/// Convert decibels back to power
/// @param S_db Decibel spectrogram
/// @param ref Reference power
/// @return Power spectrogram
ArrayXXr db_to_power(const ArrayXXr& S_db, Real ref = 1.0);
Real db_to_power(Real S_db, Real ref = 1.0);

/// Convert amplitude spectrogram to decibels
/// @param S Amplitude spectrogram
/// @param ref Reference amplitude
/// @param amin Minimum threshold
/// @param top_db Maximum dynamic range
/// @return Decibel-scaled spectrogram
ArrayXXr amplitude_to_db(const ArrayXXr& S,
                         Real ref = 1.0,
                         Real amin = 1e-5,
                         std::optional<Real> top_db = 80.0);

ArrayXXr amplitude_to_db(const ArrayXXr& S,
                         std::function<Real(const ArrayXXr&)> ref,
                         Real amin = 1e-5,
                         std::optional<Real> top_db = 80.0);

Real amplitude_to_db(Real S, Real ref = 1.0, Real amin = 1e-5,
                     std::optional<Real> top_db = std::nullopt);

/// Convert decibels back to amplitude
/// @param S_db Decibel spectrogram
/// @param ref Reference amplitude
/// @return Amplitude spectrogram
ArrayXXr db_to_amplitude(const ArrayXXr& S_db, Real ref = 1.0);
Real db_to_amplitude(Real S_db, Real ref = 1.0);

// ============================================================================
// Perceptual Weighting
// ============================================================================

/// Apply perceptual frequency weighting to a spectrogram
/// @param S Power spectrogram
/// @param frequencies Frequency bins in Hz
/// @param kind Weighting type (A, B, C, D, Z)
/// @param ref Reference power
/// @param amin Minimum amplitude
/// @param top_db Maximum dynamic range
/// @return Perceptually weighted spectrogram
ArrayXXr perceptual_weighting(const ArrayXXr& S,
                              const ArrayXr& frequencies,
                              WeightType kind = WeightType::A,
                              Real ref = 1.0,
                              Real amin = 1e-10,
                              std::optional<Real> top_db = 80.0);

// ============================================================================
// Phase Vocoder
// ============================================================================

/// Time-stretch a spectrogram using phase vocoder
/// @param D Complex STFT matrix
/// @param rate Stretch factor (>1 speeds up, <1 slows down)
/// @param hop_length Original hop length
/// @param n_fft FFT size (inferred if nullopt)
/// @return Time-stretched STFT matrix
ArrayXXc phase_vocoder(const ArrayXXc& D,
                       Real rate,
                       std::optional<int> hop_length = std::nullopt,
                       std::optional<int> n_fft = std::nullopt);

// ============================================================================
// Griffin-Lim Algorithm
// ============================================================================

/// Reconstruct audio from magnitude spectrogram using Griffin-Lim
/// @param S Magnitude spectrogram
/// @param n_iter Number of iterations
/// @param hop_length Samples between frames
/// @param win_length Window length
/// @param n_fft FFT size
/// @param window Window type
/// @param center Whether to center frames
/// @param length Desired output length
/// @param pad_mode Padding mode
/// @param momentum Griffin-Lim momentum (0 to 1)
/// @param init_phase Initial phase estimate ('random' or 'zeros')
/// @param random_state Random seed for initial phase
/// @return Reconstructed time-domain signal
ArrayXr griffinlim(const ArrayXXr& S,
                   int n_iter = 32,
                   std::optional<int> hop_length = std::nullopt,
                   std::optional<int> win_length = std::nullopt,
                   std::optional<int> n_fft = std::nullopt,
                   WindowType window = WindowType::Hann,
                   bool center = true,
                   std::optional<int> length = std::nullopt,
                   PadMode pad_mode = PadMode::Constant,
                   Real momentum = 0.99,
                   const std::string& init_phase = "random",
                   std::optional<unsigned int> random_state = std::nullopt);

// ============================================================================
// PCEN (Per-Channel Energy Normalization)
// ============================================================================

/// Apply per-channel energy normalization
/// @param S Input spectrogram (power or magnitude)
/// @param sr Sample rate
/// @param hop_length Hop length used to compute S
/// @param gain Gain factor
/// @param bias Bias added to denominator
/// @param power Compression power
/// @param time_constant Smoothing time constant
/// @param eps Stability constant
/// @return Normalized spectrogram
ArrayXXr pcen(const ArrayXXr& S,
              Real sr = 22050,
              int hop_length = 512,
              Real gain = 0.98,
              Real bias = 2.0,
              Real power = 0.5,
              Real time_constant = 0.4,
              Real eps = 1e-6);

// ============================================================================
// Window Functions
// ============================================================================

/// Get a window function
/// @param window_type Window type
/// @param n_fft Window length
/// @param fftbins If true, create symmetric window
/// @return Window function values
ArrayXr get_window(WindowType window_type, int n_fft, bool fftbins = true);

/// Get a window function from string name
/// @param window_name Window name ("hann", "hamming", etc.)
/// @param n_fft Window length
/// @param fftbins If true, create symmetric window
/// @return Window function values
ArrayXr get_window(const std::string& window_name, int n_fft, bool fftbins = true);

/// Compute the sum of squared window values at each output sample
/// @param window Window function values
/// @param n_frames Number of frames
/// @param hop_length Hop length
/// @param n_fft FFT size
/// @return Sum of squared windows at each output sample
ArrayXr window_sumsquare(const ArrayXr& window,
                         int n_frames,
                         int hop_length,
                         int n_fft);

// ============================================================================
// IIR Time-Frequency Representation
// ============================================================================

/// Time-frequency representation using IIR filters (semitone filterbank)
/// @param y Audio time series
/// @param sr Sample rate
/// @param win_length Window length for STMSP
/// @param hop_length Hop length (default: win_length / 4)
/// @param center If true, center frames
/// @param tuning Tuning deviation from A440
/// @param pad_mode Padding mode
/// @return Short-time mean-square power (n_filters, n_frames)
ArrayXXr iirt(const ArrayXr& y, Real sr = 22050, int win_length = 2048,
              std::optional<int> hop_length = std::nullopt,
              bool center = true, Real tuning = 0.0,
              PadMode pad_mode = PadMode::Constant);

// ============================================================================
// Reassigned Spectrogram
// ============================================================================

/// Time-frequency reassigned spectrogram
/// @param y Audio time series
/// @param sr Sample rate
/// @param n_fft FFT size
/// @param hop_length Hop length
/// @param win_length Window length
/// @param window Window type
/// @param center If true, center frames
/// @param reassign_frequencies If true, compute reassigned frequencies
/// @param reassign_times If true, compute reassigned times
/// @param ref_power Minimum power threshold for reassignment
/// @param fill_nan If true, fill NaN with bin center values
/// @param clip If true, clip to valid ranges
/// @param pad_mode Padding mode
/// @return Tuple of (frequencies, times, magnitudes)
std::tuple<ArrayXXr, ArrayXXr, ArrayXXr>
reassigned_spectrogram(const ArrayXr& y, Real sr = 22050, int n_fft = 2048,
                       std::optional<int> hop_length = std::nullopt,
                       std::optional<int> win_length = std::nullopt,
                       WindowType window = WindowType::Hann,
                       bool center = true,
                       bool reassign_frequencies = true,
                       bool reassign_times = true,
                       Real ref_power = 1e-6,
                       bool fill_nan = false, bool clip = true,
                       PadMode pad_mode = PadMode::Constant);

// ============================================================================
// Fast Mellin Transform
// ============================================================================

/// Fast Mellin Transform (scale transform)
/// @param y Input signal
/// @param t_min Minimum time spacing (in samples)
/// @param n_fmt Number of scale transform bins (nullopt = auto)
/// @param kind Interpolation type ("cubic" or "linear")
/// @param beta Mellin parameter (0.5 for scale transform)
/// @param over_sample Over-sampling factor
/// @return Complex scale transform coefficients
ArrayXc fmt(const ArrayXr& y, Real t_min = 0.5,
            std::optional<int> n_fmt = std::nullopt,
            const std::string& kind = "cubic",
            Real beta = 0.5, Real over_sample = 1.0);

} // namespace librosa
