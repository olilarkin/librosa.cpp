#pragma once

#include "../types.hpp"
#include <optional>
#include <string>

namespace librosa {

// ============================================================================
// Constant-Q Transform
// ============================================================================

/// Compute the constant-Q transform of an audio signal.
/// CQT is a special case of VQT with gamma=0.
/// @param y Audio time series (mono)
/// @param sr Sample rate
/// @param hop_length Samples between successive CQT columns
/// @param fmin Minimum frequency (nullopt = C1 ~32.70 Hz)
/// @param n_bins Number of frequency bins
/// @param bins_per_octave Number of bins per octave
/// @param tuning Tuning offset in fractions of a bin (nullopt = auto-estimate)
/// @param filter_scale Filter scale factor
/// @param norm Normalization type for basis functions (nullopt = none)
/// @param sparsity Sparsification quantile for CQT basis
/// @param window Window type for basis filters
/// @param scale If true, scale CQT by sqrt(filter lengths)
/// @param pad_mode Padding mode for centered frame analysis
/// @return Complex CQT matrix (n_bins, n_frames)
ArrayXXc cqt(const ArrayXr& y,
             Real sr = 22050,
             int hop_length = 512,
             std::optional<Real> fmin = std::nullopt,
             int n_bins = 84,
             int bins_per_octave = 12,
             std::optional<Real> tuning = 0.0,
             Real filter_scale = 1.0,
             std::optional<Real> norm = 1.0,
             Real sparsity = 0.01,
             WindowType window = WindowType::Hann,
             bool scale = true,
             PadMode pad_mode = PadMode::Constant);

/// Compute the variable-Q transform of an audio signal.
/// Uses recursive sub-sampling for multi-octave processing.
/// @param y Audio time series (mono)
/// @param sr Sample rate
/// @param hop_length Samples between successive VQT columns
/// @param fmin Minimum frequency (nullopt = C1 ~32.70 Hz)
/// @param n_bins Number of frequency bins
/// @param gamma Bandwidth offset (0 = CQT, nullopt = ERB-proportional)
/// @param bins_per_octave Number of bins per octave
/// @param tuning Tuning offset (nullopt = auto-estimate)
/// @param filter_scale Filter scale factor
/// @param norm Normalization type for basis functions
/// @param sparsity Sparsification quantile for VQT basis
/// @param window Window type for basis filters
/// @param scale If true, scale VQT by sqrt(filter lengths)
/// @param pad_mode Padding mode
/// @return Complex VQT matrix (n_bins, n_frames)
ArrayXXc vqt(const ArrayXr& y,
             Real sr = 22050,
             int hop_length = 512,
             std::optional<Real> fmin = std::nullopt,
             int n_bins = 84,
             std::optional<Real> gamma = std::nullopt,
             int bins_per_octave = 12,
             std::optional<Real> tuning = 0.0,
             Real filter_scale = 1.0,
             std::optional<Real> norm = 1.0,
             Real sparsity = 0.01,
             WindowType window = WindowType::Hann,
             bool scale = true,
             PadMode pad_mode = PadMode::Constant);

/// Compute the pseudo constant-Q transform (magnitude only, single FFT size).
/// @param y Audio time series (mono)
/// @param sr Sample rate
/// @param hop_length Samples between successive columns
/// @param fmin Minimum frequency (nullopt = C1)
/// @param n_bins Number of frequency bins
/// @param bins_per_octave Number of bins per octave
/// @param tuning Tuning offset (nullopt = auto-estimate)
/// @param filter_scale Filter scale factor
/// @param norm Normalization type for basis functions
/// @param sparsity Sparsification quantile
/// @param window Window type
/// @param scale If true, scale by sqrt(n_fft)
/// @param pad_mode Padding mode
/// @return Real pseudo-CQT matrix (n_bins, n_frames)
ArrayXXr pseudo_cqt(const ArrayXr& y,
                    Real sr = 22050,
                    int hop_length = 512,
                    std::optional<Real> fmin = std::nullopt,
                    int n_bins = 84,
                    int bins_per_octave = 12,
                    std::optional<Real> tuning = 0.0,
                    Real filter_scale = 1.0,
                    std::optional<Real> norm = 1.0,
                    Real sparsity = 0.01,
                    WindowType window = WindowType::Hann,
                    bool scale = true,
                    PadMode pad_mode = PadMode::Constant);

/// Compute the inverse constant-Q transform.
/// @param C CQT representation (n_bins, n_frames)
/// @param sr Sample rate
/// @param hop_length Samples between successive frames
/// @param fmin Minimum frequency (nullopt = C1)
/// @param bins_per_octave Number of bins per octave
/// @param tuning Tuning offset
/// @param filter_scale Filter scale factor
/// @param norm Normalization type for basis functions
/// @param sparsity Sparsification quantile
/// @param window Window type
/// @param scale If true, the forward CQT was scaled by sqrt(filter lengths)
/// @param length Desired output length in samples (nullopt = automatic)
/// @return Reconstructed audio signal
ArrayXr icqt(const ArrayXXc& C,
             Real sr = 22050,
             int hop_length = 512,
             std::optional<Real> fmin = std::nullopt,
             int bins_per_octave = 12,
             Real tuning = 0.0,
             Real filter_scale = 1.0,
             std::optional<Real> norm = 1.0,
             Real sparsity = 0.01,
             WindowType window = WindowType::Hann,
             bool scale = true,
             std::optional<int> length = std::nullopt);

/// Compute the hybrid constant-Q transform of an audio signal.
/// Uses pseudo CQT for higher frequencies where the hop_length is longer
/// than half the filter length, and the full CQT for lower frequencies.
/// @param y Audio time series (mono)
/// @param sr Sample rate
/// @param hop_length Samples between successive CQT columns
/// @param fmin Minimum frequency (nullopt = C1 ~32.70 Hz)
/// @param n_bins Number of frequency bins
/// @param bins_per_octave Number of bins per octave
/// @param tuning Tuning offset in fractions of a bin (nullopt = auto-estimate)
/// @param filter_scale Filter scale factor
/// @param norm Normalization type for basis functions (nullopt = none)
/// @param sparsity Sparsification quantile for CQT basis
/// @param window Window type for basis filters
/// @param scale If true, scale CQT by sqrt(filter lengths)
/// @param pad_mode Padding mode for centered frame analysis
/// @return Real CQT magnitude matrix (n_bins, n_frames)
ArrayXXr hybrid_cqt(const ArrayXr& y,
                     Real sr = 22050,
                     int hop_length = 512,
                     std::optional<Real> fmin = std::nullopt,
                     int n_bins = 84,
                     int bins_per_octave = 12,
                     std::optional<Real> tuning = 0.0,
                     Real filter_scale = 1.0,
                     std::optional<Real> norm = 1.0,
                     Real sparsity = 0.01,
                     WindowType window = WindowType::Hann,
                     bool scale = true,
                     PadMode pad_mode = PadMode::Constant);

/// Reconstruct audio from CQT magnitude using Griffin-Lim algorithm.
/// @param C CQT magnitude spectrogram (n_bins, n_frames)
/// @param n_iter Number of Griffin-Lim iterations
/// @param sr Sample rate
/// @param hop_length Samples between frames
/// @param fmin Minimum frequency (nullopt = C1)
/// @param bins_per_octave Number of bins per octave
/// @param tuning Tuning offset
/// @param filter_scale Filter scale factor
/// @param norm Normalization type for basis functions
/// @param sparsity Sparsification quantile
/// @param window Window type
/// @param scale If true, scale CQT by sqrt(filter lengths)
/// @param pad_mode Padding mode
/// @param length Desired output length
/// @param momentum Griffin-Lim momentum (0 to 1)
/// @param init_phase Phase initialization ("random" or "ones")
/// @param random_state Random seed for initial phase
/// @return Reconstructed audio signal
ArrayXr griffinlim_cqt(const ArrayXXr& C,
                       int n_iter = 32,
                       Real sr = 22050,
                       int hop_length = 512,
                       std::optional<Real> fmin = std::nullopt,
                       int bins_per_octave = 12,
                       Real tuning = 0.0,
                       Real filter_scale = 1.0,
                       std::optional<Real> norm = 1.0,
                       Real sparsity = 0.01,
                       WindowType window = WindowType::Hann,
                       bool scale = true,
                       PadMode pad_mode = PadMode::Constant,
                       std::optional<int> length = std::nullopt,
                       Real momentum = 0.99,
                       const std::string& init_phase = "random",
                       std::optional<unsigned int> random_state = std::nullopt);

} // namespace librosa
