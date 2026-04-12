#pragma once

#include "../types.hpp"
#include <vector>

namespace librosa {
namespace core {

/// Interpolate a spectrogram at harmonic multiples of each frequency bin
///
/// For each harmonic h in harmonics, computes target_freqs = h * freqs and
/// interpolates x at those frequencies using linear interpolation.
///
/// @param x 2D spectrogram [shape: (n_freq, n_frames)]
/// @param freqs Frequency values for each row of x [shape: (n_freq,)]
/// @param harmonics Harmonic multipliers (e.g., {1, 2, 3} for fundamental + 2nd + 3rd)
/// @param fill_value Value for out-of-range interpolation
/// @return Stacked harmonic interpolation [shape: (n_harmonics * n_freq, n_frames)]
ArrayXXr interp_harmonics(const ArrayXXr& x, const ArrayXr& freqs,
                          const std::vector<Real>& harmonics,
                          Real fill_value = 0);

/// Interpolate a 1D spectrum at harmonic multiples
/// @param x 1D spectrum [shape: (n_freq,)]
/// @param freqs Frequency values [shape: (n_freq,)]
/// @param harmonics Harmonic multipliers
/// @param fill_value Value for out-of-range interpolation
/// @return Stacked harmonic interpolation [shape: (n_harmonics, n_freq)]
ArrayXXr interp_harmonics(const ArrayXr& x, const ArrayXr& freqs,
                          const std::vector<Real>& harmonics,
                          Real fill_value = 0);

/// Extract spectrogram energy at harmonics of a time-varying fundamental frequency
///
/// For each frame t, interpolates x[:,t] at frequencies harmonics[i] * f0[t]
///
/// @param x 2D spectrogram [shape: (n_freq, n_frames)]
/// @param f0 Fundamental frequency per frame [shape: (n_frames,)]
/// @param freqs Frequency values for each row of x [shape: (n_freq,)]
/// @param harmonics Harmonic multipliers
/// @param fill_value Value for out-of-range interpolation
/// @return Harmonic energies [shape: (n_harmonics, n_frames)]
ArrayXXr f0_harmonics(const ArrayXXr& x, const ArrayXr& f0,
                      const ArrayXr& freqs,
                      const std::vector<Real>& harmonics,
                      Real fill_value = 0);

/// Compute harmonic salience of a spectrogram
///
/// Sums weighted energy at harmonics of each frequency, optionally filtering peaks.
///
/// @param S Input spectrogram (magnitude or power) [shape: (n_freq, n_frames)]
/// @param freqs Frequency values [shape: (n_freq,)]
/// @param harmonics Harmonic multipliers
/// @param weights Weights for each harmonic (empty = uniform)
/// @param filter_peaks If true, only keep local maxima in frequency
/// @param fill_value Value for out-of-range interpolation
/// @return Salience function [shape: (n_freq, n_frames)]
ArrayXXr salience(const ArrayXXr& S, const ArrayXr& freqs,
                  const std::vector<Real>& harmonics,
                  const std::vector<Real>& weights = {},
                  bool filter_peaks = true, Real fill_value = 0);

} // namespace core
} // namespace librosa
