#pragma once

#include "../types.hpp"
#include <optional>
#include <vector>

namespace librosa {
namespace feature {

/// Compute the Fourier tempogram: short-time Fourier transform of the
/// onset strength envelope.
///
/// @param onset_envelope Pre-computed onset strength envelope
/// @param sr Sample rate
/// @param hop_length Hop length used for onset envelope
/// @param win_length Length of the FFT window (in frames)
/// @param center If true, center the analysis windows
/// @param window Window type
/// @return Complex Fourier tempogram [shape: (win_length/2+1, n_frames)]
ArrayXXc fourier_tempogram(const ArrayXr& onset_envelope,
                           Real sr = 22050, int hop_length = 512,
                           int win_length = 384, bool center = true,
                           WindowType window = WindowType::Hann);

/// Compute Fourier tempogram from audio
/// @param y Audio time series
/// @param sr Sample rate
/// @param hop_length Hop length
/// @param win_length Length of the FFT window
/// @param center If true, center the analysis windows
/// @param window Window type
/// @return Complex Fourier tempogram
ArrayXXc fourier_tempogram_audio(const ArrayXr& y,
                                 Real sr = 22050, int hop_length = 512,
                                 int win_length = 384, bool center = true,
                                 WindowType window = WindowType::Hann);

/// Compute tempo ratio features from a tempogram via harmonic interpolation.
///
/// Estimates per-frame tempo and extracts energy at harmonic factors of tempo.
///
/// @param tg Autocorrelation tempogram [shape: (win_length, n_frames)]
/// @param sr Sample rate
/// @param hop_length Hop length
/// @param factors Tempo ratio factors (default: standard musical ratios)
/// @param start_bpm Initial tempo guess for prior
/// @param std_bpm Tempo prior standard deviation
/// @param max_tempo Maximum tempo to consider
/// @param fill_value Value for out-of-range interpolation
/// @return Tempogram ratio features [shape: (n_factors, n_frames)]
ArrayXXr tempogram_ratio(const ArrayXXr& tg, Real sr = 22050,
                         int hop_length = 512,
                         const std::vector<Real>& factors = {},
                         Real start_bpm = 120, Real std_bpm = 1.0,
                         std::optional<Real> max_tempo = 320.0,
                         Real fill_value = 0);

} // namespace feature
} // namespace librosa
