#pragma once

#include "types.hpp"
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace librosa {
namespace segment {

// ============================================================================
// Cross-similarity and Recurrence
// ============================================================================

/// Compute cross-similarity between two feature matrices using KNN.
/// @param data Feature matrix for comparison sequence [shape: (d, n)]
/// @param data_ref Feature matrix for reference sequence [shape: (d, n_ref)]
/// @param k Number of nearest neighbors (default: 2*ceil(sqrt(n_ref)))
/// @param metric Distance metric: "euclidean" or "cosine"
/// @param mode Output mode: "connectivity", "distance", or "affinity"
/// @param bandwidth Affinity bandwidth (only for mode="affinity"; 0 = auto)
/// @return Cross-similarity matrix [shape: (n_ref, n)]
ArrayXXr cross_similarity(
    const ArrayXXr& data,
    const ArrayXXr& data_ref,
    int k = 0,
    const std::string& metric = "euclidean",
    const std::string& mode = "connectivity",
    Real bandwidth = 0.0);

/// Sparse version of cross_similarity (returns SparseMatrixXr).
SparseMatrixXr cross_similarity_sparse(
    const ArrayXXr& data,
    const ArrayXXr& data_ref,
    int k = 0,
    const std::string& metric = "euclidean",
    const std::string& mode = "connectivity",
    Real bandwidth = 0.0);

/// Compute a recurrence (self-similarity) matrix from a feature matrix.
/// @param data Feature matrix [shape: (d, n)]
/// @param k Number of nearest neighbors (default: 2*ceil(sqrt(n - 2*width + 1)))
/// @param width Minimum lag distance (frames within +/- width are excluded)
/// @param metric Distance metric: "euclidean" or "cosine"
/// @param sym If true, require mutual nearest neighbors
/// @param mode Output mode: "connectivity", "distance", or "affinity"
/// @param bandwidth Affinity bandwidth (only for mode="affinity"; 0 = auto)
/// @param self_ If true, populate the main diagonal
/// @return Recurrence matrix [shape: (n, n)]
ArrayXXr recurrence_matrix(
    const ArrayXXr& data,
    int k = 0,
    int width = 1,
    const std::string& metric = "euclidean",
    bool sym = false,
    const std::string& mode = "connectivity",
    Real bandwidth = 0.0,
    bool self_ = false);

/// Sparse version of recurrence_matrix.
SparseMatrixXr recurrence_matrix_sparse(
    const ArrayXXr& data,
    int k = 0,
    int width = 1,
    const std::string& metric = "euclidean",
    bool sym = false,
    const std::string& mode = "connectivity",
    Real bandwidth = 0.0,
    bool self_ = false);

// ============================================================================
// Lag / Recurrence Conversion
// ============================================================================

/// Convert a recurrence matrix to a lag matrix.
/// lag[i, j] == rec[i+j, j]
/// @param rec Square recurrence matrix [shape: (n, n)]
/// @param pad If true, pad with n zeros to eliminate repetition assumption
/// @param axis Time axis (0 or 1; default -1 = 1)
/// @return Lag matrix
ArrayXXr recurrence_to_lag(const ArrayXXr& rec, bool pad = true, int axis = -1);

/// Convert a lag matrix back to a recurrence matrix.
/// @param lag Lag matrix
/// @param axis Time axis (0 or 1; default -1 = 1)
/// @return Recurrence matrix [shape: (n, n)]
ArrayXXr lag_to_recurrence(const ArrayXXr& lag, int axis = -1);

// ============================================================================
// Timelag Filter
// ============================================================================

/// Apply a filter function in the time-lag domain.
/// Equivalent to: rec_to_lag -> apply filter -> lag_to_rec
/// @param rec Recurrence matrix [shape: (n, n)]
/// @param filter_fn Filter function that takes an ArrayXXr and returns ArrayXXr
/// @param pad Whether to zero-pad the lag representation
/// @return Filtered recurrence matrix
template<typename FilterFunc>
ArrayXXr timelag_filter(const ArrayXXr& rec, FilterFunc filter_fn, bool pad = true) {
    ArrayXXr lag = recurrence_to_lag(rec, pad);
    ArrayXXr filtered = filter_fn(lag);
    return lag_to_recurrence(filtered);
}

// ============================================================================
// Path Enhancement
// ============================================================================

/// Multi-angle path enhancement for similarity matrices.
/// Convolves multiple diagonal smoothing filters and takes element-wise max.
/// @param R Self- or cross-similarity matrix [shape: (n, n)]
/// @param n Filter length
/// @param window Window type for smoothing filters
/// @param max_ratio Maximum tempo ratio
/// @param min_ratio Minimum tempo ratio (default: 1/max_ratio)
/// @param n_filters Number of smoothing filters
/// @param zero_mean If true, create zero-mean filters
/// @param clip If true, clip output to non-negative
/// @return Smoothed similarity matrix
ArrayXXr path_enhance(
    const ArrayXXr& R,
    int n,
    WindowType window = WindowType::Hann,
    Real max_ratio = 2.0,
    Real min_ratio = 0.0,
    int n_filters = 7,
    bool zero_mean = false,
    bool clip = true);

// ============================================================================
// Temporal Clustering
// ============================================================================

/// Bottom-up temporal segmentation using constrained agglomerative clustering.
/// @param data Feature matrix [shape: (d, n)] — features x time
/// @param k Number of segments to produce
/// @param axis Axis along which to cluster (default: -1, last axis)
/// @return Sorted vector of left-boundary frame indices (always includes 0)
std::vector<Eigen::Index> agglomerative(
    const ArrayXXr& data,
    int k,
    int axis = -1);

/// Sub-divide a segmentation by feature clustering.
/// Each interval defined by frames is partitioned into n_segments.
/// @param data Feature matrix [shape: (d, n)]
/// @param frames Array of boundary frame indices
/// @param n_segments Maximum number of sub-segments per interval
/// @param axis Axis along which to segment
/// @return Vector of sub-divided boundary indices
std::vector<Eigen::Index> subsegment(
    const ArrayXXr& data,
    const std::vector<Eigen::Index>& frames,
    int n_segments = 4,
    int axis = -1);

} // namespace segment
} // namespace librosa
