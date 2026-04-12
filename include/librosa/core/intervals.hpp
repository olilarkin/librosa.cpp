#pragma once

#include "../types.hpp"
#include <string>
#include <vector>

namespace librosa {

// ============================================================================
// Interval Construction
// ============================================================================

/// Construct Pythagorean intervals by stacking perfect fifths
/// @param bins_per_octave Number of intervals to generate
/// @param sort If true, return in ascending order; else circle-of-fifths order
/// @return Array of intervals in [1, 2)
ArrayXr pythagorean_intervals(int bins_per_octave = 12, bool sort = true);

/// Construct p-limit intervals for a given set of prime factors
/// Uses harmonic crystal growth algorithm.
/// @param primes Vector of odd primes (e.g., {3} for 3-limit, {3,5} for 5-limit)
/// @param bins_per_octave Number of intervals to construct
/// @param sort If true, return in ascending order
/// @return Array of intervals in [1, 2)
ArrayXr plimit_intervals(const std::vector<int>& primes,
                         int bins_per_octave = 12,
                         bool sort = true);

/// Construct a set of frequencies from an interval specification
/// @param n_bins Number of frequencies to generate
/// @param fmin Minimum frequency
/// @param intervals Interval type: "equal", "pythagorean", "ji3", "ji5", "ji7"
/// @param bins_per_octave Bins per octave (for string interval types)
/// @param tuning Tuning deviation in fractional bins (only for "equal")
/// @param sort Sort intervals in ascending order
/// @return Array of frequencies
ArrayXr interval_frequencies(int n_bins,
                             Real fmin,
                             const std::string& intervals,
                             int bins_per_octave = 12,
                             Real tuning = 0.0,
                             bool sort = true);

/// Construct frequencies from explicit interval ratios
/// @param n_bins Number of frequencies to generate
/// @param fmin Minimum frequency
/// @param intervals Array of interval ratios in [1, 2)
/// @param sort Sort in ascending order
/// @return Array of frequencies
ArrayXr interval_frequencies(int n_bins,
                             Real fmin,
                             const ArrayXr& intervals,
                             bool sort = true);

} // namespace librosa
