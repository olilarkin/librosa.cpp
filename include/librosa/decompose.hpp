#pragma once

#include "types.hpp"
#include <utility>
#include <variant>

namespace librosa {
namespace decompose {

// ============================================================================
// Median Filter (internal utility)
// ============================================================================

/// Apply a median filter to a 2D array
/// @param S Input array [shape=(rows, cols)]
/// @param size Filter size (rows, cols)
/// @param mode Padding mode: "reflect", "constant", "edge"
/// @return Filtered array
ArrayXXr median_filter_2d(
    const ArrayXXr& S,
    std::pair<int, int> size,
    const std::string& mode = "reflect");

// ============================================================================
// HPSS - Harmonic/Percussive Source Separation
// ============================================================================

/// Median-filtering harmonic percussive source separation (HPSS)
///
/// If margin = 1.0, decomposes an input spectrogram S = H + P
/// where H contains harmonic components and P contains percussive components.
///
/// If margin > 1.0, decomposes S = H + P + R where R is residual.
///
/// Based on Fitzgerald (2010) and Driedger et al. (2014)
///
/// @param S Input spectrogram (magnitude or complex)
/// @param kernel_size Median filter kernel size(s)
///        If single value, used for both. If pair, (harmonic, percussive)
/// @param power Exponent for Wiener filter when constructing soft masks
/// @param mask If true, return masks instead of components
/// @param margin Margin size(s) for masks. If pair, (margin_harm, margin_perc)
/// @return Pair of (harmonic, percussive) components or masks
std::pair<ArrayXXr, ArrayXXr> hpss(
    const ArrayXXr& S,
    int kernel_size = 31,
    Real power = 2.0,
    bool mask = false,
    Real margin = 1.0);

/// HPSS with separate kernel sizes
std::pair<ArrayXXr, ArrayXXr> hpss(
    const ArrayXXr& S,
    std::pair<int, int> kernel_size,
    Real power = 2.0,
    bool mask = false,
    Real margin = 1.0);

/// HPSS with separate margins
std::pair<ArrayXXr, ArrayXXr> hpss(
    const ArrayXXr& S,
    int kernel_size,
    Real power,
    bool mask,
    std::pair<Real, Real> margin);

/// HPSS with separate kernel sizes and margins
std::pair<ArrayXXr, ArrayXXr> hpss(
    const ArrayXXr& S,
    std::pair<int, int> kernel_size,
    Real power,
    bool mask,
    std::pair<Real, Real> margin);

/// Complex HPSS - separates magnitude and preserves phase
std::pair<ArrayXXc, ArrayXXc> hpss_complex(
    const ArrayXXc& S,
    int kernel_size = 31,
    Real power = 2.0,
    bool mask = false,
    Real margin = 1.0);

/// Complex HPSS with separate kernel sizes
std::pair<ArrayXXc, ArrayXXc> hpss_complex(
    const ArrayXXc& S,
    std::pair<int, int> kernel_size,
    Real power = 2.0,
    bool mask = false,
    Real margin = 1.0);

/// Complex HPSS with separate margins
std::pair<ArrayXXc, ArrayXXc> hpss_complex(
    const ArrayXXc& S,
    int kernel_size,
    Real power,
    bool mask,
    std::pair<Real, Real> margin);

/// Complex HPSS with all parameters
std::pair<ArrayXXc, ArrayXXc> hpss_complex(
    const ArrayXXc& S,
    std::pair<int, int> kernel_size,
    Real power,
    bool mask,
    std::pair<Real, Real> margin);

// ============================================================================
// NMF Decomposition
// ============================================================================

/// Non-negative matrix factorization (NMF) decomposition.
/// Decomposes S ≈ components * activations where both factors are non-negative.
/// @param S Input feature matrix [shape: (n_features, n_samples)], non-negative
/// @param n_components Number of components (0 = n_features)
/// @param sort If true, sort components by ascending peak frequency
/// @param max_iter Maximum number of NMF iterations
/// @param tol Convergence tolerance
/// @return Pair of (components [n_features, n_components], activations [n_components, n_samples])
std::pair<ArrayXXr, ArrayXXr> decompose_nmf(
    const ArrayXXr& S,
    int n_components = 0,
    bool sort = false,
    int max_iter = 200,
    Real tol = 1e-4);

// ============================================================================
// Nearest-Neighbor Filter
// ============================================================================

/// Filter a feature matrix by nearest-neighbor aggregation.
/// Each column (or row if axis=0) is replaced by the aggregate of its
/// nearest neighbors as defined by a recurrence matrix.
/// @param S Input feature matrix [shape: (d, n)]
/// @param rec Sparse recurrence matrix [shape: (n, n)] (CSR format)
/// @param aggregate Aggregation function (Mean or Median)
/// @param axis Axis along which to filter (default: -1 = columns)
/// @return Filtered feature matrix
ArrayXXr nn_filter(
    const ArrayXXr& S,
    const SparseMatrixXr& rec,
    AggregateFunc aggregate = AggregateFunc::Mean,
    int axis = -1);

} // namespace decompose
} // namespace librosa
