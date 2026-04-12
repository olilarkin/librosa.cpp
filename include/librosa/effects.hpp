#pragma once

#include "types.hpp"
#include <optional>
#include <vector>
#include <utility>

namespace librosa {
namespace effects {

// ============================================================================
// Time Stretching
// ============================================================================

/// Time-stretch an audio series by a fixed rate
/// @param y Audio time series
/// @param rate Stretch factor (>1 speeds up, <1 slows down)
/// @param n_fft FFT window size
/// @param hop_length Samples between frames
/// @param win_length Window length
/// @param window Window type
/// @param center Center frames
/// @return Time-stretched audio
ArrayXr time_stretch(
    const ArrayXr& y,
    Real rate,
    int n_fft = 2048,
    std::optional<int> hop_length = std::nullopt,
    std::optional<int> win_length = std::nullopt,
    WindowType window = WindowType::Hann,
    bool center = true);

// ============================================================================
// Pitch Shifting
// ============================================================================

/// Shift the pitch of a waveform by n_steps steps
/// @param y Audio time series
/// @param sr Sample rate
/// @param n_steps Number of steps to shift (can be fractional)
/// @param bins_per_octave Number of steps per octave
/// @param res_type Resampling method ("kaiser_*", "fft", or "linear")
/// @param n_fft FFT window size
/// @param hop_length Samples between frames
/// @return Pitch-shifted audio
ArrayXr pitch_shift(
    const ArrayXr& y,
    Real sr,
    Real n_steps,
    int bins_per_octave = 12,
    const std::string& res_type = "kaiser_hq",
    int n_fft = 2048,
    std::optional<int> hop_length = std::nullopt);

// ============================================================================
// Trimming and Splitting
// ============================================================================

/// Trim leading and trailing silence from an audio signal
/// @param y Audio signal
/// @param top_db Threshold in dB below reference to consider as silence
/// @param ref Reference amplitude (max amplitude if negative)
/// @param frame_length Analysis frame length
/// @param hop_length Samples between frames
/// @return Pair of (trimmed audio, [start, end] indices)
std::pair<ArrayXr, std::pair<Eigen::Index, Eigen::Index>> trim(
    const ArrayXr& y,
    Real top_db = 60,
    Real ref = -1,  // Negative means use max
    int frame_length = 2048,
    int hop_length = 512);

/// Split an audio signal into non-silent intervals
/// @param y Audio signal
/// @param top_db Threshold in dB below reference to consider as silence
/// @param ref Reference amplitude (max amplitude if negative)
/// @param frame_length Analysis frame length
/// @param hop_length Samples between frames
/// @return Vector of (start, end) sample indices for non-silent intervals
std::vector<std::pair<Eigen::Index, Eigen::Index>> split(
    const ArrayXr& y,
    Real top_db = 60,
    Real ref = -1,
    int frame_length = 2048,
    int hop_length = 512);

// ============================================================================
// Preemphasis/Deemphasis
// ============================================================================

/// Apply preemphasis filter to audio signal
/// @param y Input audio
/// @param coef Preemphasis coefficient
/// @return Preemphasized signal
ArrayXr preemphasis(const ArrayXr& y, Real coef = 0.97);

/// Apply deemphasis filter (inverse of preemphasis)
/// @param y Preemphasized audio
/// @param coef Preemphasis coefficient
/// @return Deemphasized signal
ArrayXr deemphasis(const ArrayXr& y, Real coef = 0.97);

// ============================================================================
// Remix
// ============================================================================

/// Remix an audio signal by re-ordering time intervals
/// @param y Audio time series
/// @param intervals Vector of (start, end) sample pairs
/// @param align_zeros Map interval boundaries to zero-crossings
/// @return Remixed audio
ArrayXr remix(
    const ArrayXr& y,
    const std::vector<std::pair<Eigen::Index, Eigen::Index>>& intervals,
    bool align_zeros = true);

// ============================================================================
// Harmonic/Percussive Separation Effects
// ============================================================================

/// Extract the harmonic component of an audio signal
/// Computes STFT → HPSS → iSTFT, returning the harmonic part
/// @param y Audio time series
/// @param kernel_size Median filter kernel size for HPSS
/// @param power Exponent for Wiener filter
/// @param mask If true, use soft masks
/// @param margin Margin for HPSS masks
/// @param n_fft FFT window size
/// @param hop_length Samples between frames
/// @param win_length Window length
/// @param window Window type
/// @param center Center frames
/// @param pad_mode Padding mode
/// @return Harmonic component audio
ArrayXr harmonic(
    const ArrayXr& y,
    int kernel_size = 31,
    Real power = 2.0,
    bool mask = false,
    Real margin = 1.0,
    int n_fft = 2048,
    std::optional<int> hop_length = std::nullopt,
    std::optional<int> win_length = std::nullopt,
    WindowType window = WindowType::Hann,
    bool center = true,
    PadMode pad_mode = PadMode::Constant);

/// Extract the percussive component of an audio signal
/// Computes STFT → HPSS → iSTFT, returning the percussive part
/// @param y Audio time series
/// @param kernel_size Median filter kernel size for HPSS
/// @param power Exponent for Wiener filter
/// @param mask If true, use soft masks
/// @param margin Margin for HPSS masks
/// @param n_fft FFT window size
/// @param hop_length Samples between frames
/// @param win_length Window length
/// @param window Window type
/// @param center Center frames
/// @param pad_mode Padding mode
/// @return Percussive component audio
ArrayXr percussive(
    const ArrayXr& y,
    int kernel_size = 31,
    Real power = 2.0,
    bool mask = false,
    Real margin = 1.0,
    int n_fft = 2048,
    std::optional<int> hop_length = std::nullopt,
    std::optional<int> win_length = std::nullopt,
    WindowType window = WindowType::Hann,
    bool center = true,
    PadMode pad_mode = PadMode::Constant);

} // namespace effects
} // namespace librosa
