#include <librosa/segment.hpp>
#include <librosa/filters.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include "internal/knn.hpp"
#include "internal/convolve2d.hpp"
#include "internal/ward.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace librosa {
namespace segment {

// ============================================================================
// Internal helpers
// ============================================================================

namespace {

/// Compute pairwise distance matrix between rows, dispatching on metric
ArrayXXr compute_distance_rows(
    const ArrayXXr& X,
    const ArrayXXr& Y,
    const std::string& metric) {

    if (metric == "euclidean") {
        return internal::cdist_euclidean_rows(X, Y);
    } else if (metric == "cosine") {
        return internal::cdist_cosine_rows(X, Y);
    } else {
        throw ParameterError("Unsupported metric: " + metric +
                             ". Supported: 'euclidean', 'cosine'");
    }
}

/// Apply affinity transformation to distance values in a sparse matrix
void apply_affinity(SparseMatrixXr& graph, Real bandwidth) {
    for (int k = 0; k < graph.outerSize(); ++k) {
        for (SparseMatrixXr::InnerIterator it(graph, k); it; ++it) {
            if (it.value() > 0) {
                it.valueRef() = std::exp(-it.value() / bandwidth);
            }
        }
    }
}

} // anonymous namespace

// ============================================================================
// Cross-similarity
// ============================================================================

ArrayXXr cross_similarity(
    const ArrayXXr& data,
    const ArrayXXr& data_ref,
    int k,
    const std::string& metric,
    const std::string& mode,
    Real bandwidth) {

    if (mode != "connectivity" && mode != "distance" && mode != "affinity") {
        throw ParameterError("Invalid mode='" + mode +
                             "'. Must be 'connectivity', 'distance', or 'affinity'");
    }

    // data: (d, n), data_ref: (d, n_ref) — features x samples
    // Transpose to rows = samples for distance computation
    Eigen::Index n = data.cols();
    Eigen::Index n_ref = data_ref.cols();

    if (data.rows() != data_ref.rows()) {
        throw ParameterError("data and data_ref must have the same number of features");
    }

    // Default k
    if (k <= 0) {
        if (n_ref <= 3) {
            k = 2;
        } else {
            k = static_cast<int>(2 * std::ceil(std::sqrt(static_cast<Real>(n_ref))));
        }
    }
    k = std::min(k, static_cast<int>(n_ref));

    // Transpose: rows = samples, cols = features
    ArrayXXr data_T = data.matrix().transpose().array();
    ArrayXXr ref_T = data_ref.matrix().transpose().array();

    // Compute pairwise distances: (n, n_ref)
    ArrayXXr dist = compute_distance_rows(data_T, ref_T, metric);

    // Build KNN graph: for each sample in data, find k nearest in data_ref
    std::string knn_mode = (mode == "affinity") ? "distance" : mode;
    SparseMatrixXr graph = internal::knn_graph(dist, k, knn_mode);

    if (mode == "affinity") {
        Real bw = bandwidth;
        if (bw <= 0) {
            bw = internal::affinity_bandwidth_med_k(graph, k);
        }
        apply_affinity(graph, bw);
    } else if (mode == "connectivity") {
        // Convert to binary
        for (int outer = 0; outer < graph.outerSize(); ++outer) {
            for (SparseMatrixXr::InnerIterator it(graph, outer); it; ++it) {
                it.valueRef() = 1.0;
            }
        }
    }

    // Transpose to (n_ref, n) and convert to dense
    SparseMatrixXr graph_T = graph.transpose();
    ArrayXXr result = ArrayXXr::Zero(n_ref, n);
    for (int outer = 0; outer < graph_T.outerSize(); ++outer) {
        for (SparseMatrixXr::InnerIterator it(graph_T, outer); it; ++it) {
            result(it.row(), it.col()) = it.value();
        }
    }

    return result;
}

SparseMatrixXr cross_similarity_sparse(
    const ArrayXXr& data,
    const ArrayXXr& data_ref,
    int k,
    const std::string& metric,
    const std::string& mode,
    Real bandwidth) {

    if (mode != "connectivity" && mode != "distance" && mode != "affinity") {
        throw ParameterError("Invalid mode='" + mode +
                             "'. Must be 'connectivity', 'distance', or 'affinity'");
    }

    Eigen::Index n = data.cols();
    Eigen::Index n_ref = data_ref.cols();

    if (data.rows() != data_ref.rows()) {
        throw ParameterError("data and data_ref must have the same number of features");
    }

    if (k <= 0) {
        if (n_ref <= 3) {
            k = 2;
        } else {
            k = static_cast<int>(2 * std::ceil(std::sqrt(static_cast<Real>(n_ref))));
        }
    }
    k = std::min(k, static_cast<int>(n_ref));

    ArrayXXr data_T = data.matrix().transpose().array();
    ArrayXXr ref_T = data_ref.matrix().transpose().array();
    ArrayXXr dist = compute_distance_rows(data_T, ref_T, metric);

    std::string knn_mode = (mode == "affinity") ? "distance" : mode;
    SparseMatrixXr graph = internal::knn_graph(dist, k, knn_mode);

    if (mode == "affinity") {
        Real bw = bandwidth;
        if (bw <= 0) {
            bw = internal::affinity_bandwidth_med_k(graph, k);
        }
        apply_affinity(graph, bw);
    } else if (mode == "connectivity") {
        for (int outer = 0; outer < graph.outerSize(); ++outer) {
            for (SparseMatrixXr::InnerIterator it(graph, outer); it; ++it) {
                it.valueRef() = 1.0;
            }
        }
    }

    return SparseMatrixXr(graph.transpose());
}

// ============================================================================
// Recurrence Matrix
// ============================================================================

ArrayXXr recurrence_matrix(
    const ArrayXXr& data,
    int k,
    int width,
    const std::string& metric,
    bool sym,
    const std::string& mode,
    Real bandwidth,
    bool self_) {

    if (mode != "connectivity" && mode != "distance" && mode != "affinity") {
        throw ParameterError("Invalid mode='" + mode +
                             "'. Must be 'connectivity', 'distance', or 'affinity'");
    }

    // data: (d, n) — features x time
    Eigen::Index t = data.cols();

    if (width < 1 || width >= (t - 1) / 2) {
        throw ParameterError("width must be at least 1 and less than (n-1)/2");
    }

    // Default k
    if (k <= 0) {
        Eigen::Index effective_t = t - 2 * width + 1;
        if (effective_t <= 3) {
            k = 2;
        } else {
            k = static_cast<int>(2 * std::ceil(std::sqrt(static_cast<Real>(effective_t))));
        }
    }

    // We need k + 2*width neighbors to ensure we have enough after removing the width exclusion zone
    int k_search = std::min(static_cast<int>(t - 1), k + 2 * width);

    // Transpose to rows = samples
    ArrayXXr data_T = data.matrix().transpose().array();

    // Self-distance
    ArrayXXr dist = compute_distance_rows(data_T, data_T, metric);

    // Build KNN graph with extra neighbors
    std::string knn_mode = (mode == "affinity") ? "distance" : mode;
    SparseMatrixXr graph = internal::knn_graph(dist, k_search, knn_mode);

    // Convert to LIL-like format for manipulation — work with dense for simplicity
    ArrayXXr rec = ArrayXXr::Zero(t, t);
    for (int outer = 0; outer < graph.outerSize(); ++outer) {
        for (SparseMatrixXr::InnerIterator it(graph, outer); it; ++it) {
            rec(it.row(), it.col()) = it.value();
        }
    }

    // Remove connections within width
    for (int diag = -(width - 1); diag <= (width - 1); ++diag) {
        for (Eigen::Index i = 0; i < t; ++i) {
            Eigen::Index j = i + diag;
            if (j >= 0 && j < t) {
                rec(i, j) = 0.0;
            }
        }
    }

    // Retain only top-k per row
    for (Eigen::Index i = 0; i < t; ++i) {
        // Collect non-zero entries
        std::vector<std::pair<Real, Eigen::Index>> entries;
        for (Eigen::Index j = 0; j < t; ++j) {
            if (rec(i, j) != 0.0) {
                entries.push_back({rec(i, j), j});
            }
        }

        if (static_cast<int>(entries.size()) > k) {
            if (mode == "connectivity") {
                // For connectivity, just keep k entries (they're all 1.0)
                // Remove from k onwards
                for (size_t idx = k; idx < entries.size(); ++idx) {
                    rec(i, entries[idx].second) = 0.0;
                }
            } else {
                // Sort by distance ascending
                std::sort(entries.begin(), entries.end());
                for (size_t idx = k; idx < entries.size(); ++idx) {
                    rec(i, entries[idx].second) = 0.0;
                }
            }
        }
    }

    // Handle self-loop diagonal
    if (self_) {
        if (mode == "connectivity") {
            for (Eigen::Index i = 0; i < t; ++i) {
                rec(i, i) = 1.0;
            }
        } else if (mode == "affinity") {
            for (Eigen::Index i = 0; i < t; ++i) {
                rec(i, i) = 1.0;  // Will become exp(0) = 1 after affinity
            }
        }
        // distance mode: diagonal is 0 (no self-distance)
    } else {
        for (Eigen::Index i = 0; i < t; ++i) {
            rec(i, i) = 0.0;
        }
    }

    // Symmetrize
    if (sym) {
        // Mutual nearest neighbors: element-wise minimum of rec and rec^T
        ArrayXXr rec_T = rec.matrix().transpose().array();
        rec = rec.min(rec_T);
    }

    // Apply affinity transformation
    if (mode == "affinity") {
        // Compute bandwidth from the distance values
        Real bw = bandwidth;
        if (bw <= 0) {
            // Median of k-th neighbor distances
            std::vector<Real> kth_dists;
            for (Eigen::Index i = 0; i < t; ++i) {
                std::vector<Real> row_dists;
                for (Eigen::Index j = 0; j < t; ++j) {
                    if (rec(i, j) > 0 && i != j) {
                        row_dists.push_back(rec(i, j));
                    }
                }
                if (!row_dists.empty()) {
                    std::sort(row_dists.begin(), row_dists.end());
                    int idx = std::min(k - 1, static_cast<int>(row_dists.size()) - 1);
                    kth_dists.push_back(row_dists[idx]);
                }
            }
            if (!kth_dists.empty()) {
                size_t mid = kth_dists.size() / 2;
                std::nth_element(kth_dists.begin(), kth_dists.begin() + mid, kth_dists.end());
                bw = kth_dists[mid];
            } else {
                bw = 1.0;
            }
        }

        for (Eigen::Index i = 0; i < t; ++i) {
            for (Eigen::Index j = 0; j < t; ++j) {
                if (rec(i, j) > 0) {
                    rec(i, j) = std::exp(-rec(i, j) / bw);
                }
            }
        }

        // Self-loops that were set to 1.0 are already correct (exp(0) = 1)
        if (self_) {
            for (Eigen::Index i = 0; i < t; ++i) {
                rec(i, i) = 1.0;
            }
        }
    } else if (mode == "connectivity") {
        // Ensure boolean-like values
        rec = (rec > 0).cast<Real>();
    }

    // Transpose to column-major convention (matching Python librosa)
    return rec.matrix().transpose().array();
}

SparseMatrixXr recurrence_matrix_sparse(
    const ArrayXXr& data,
    int k,
    int width,
    const std::string& metric,
    bool sym,
    const std::string& mode,
    Real bandwidth,
    bool self_) {

    // Compute dense and convert to sparse
    ArrayXXr dense = recurrence_matrix(data, k, width, metric, sym, mode, bandwidth, self_);

    Eigen::Index t = dense.rows();
    using Triplet = Eigen::Triplet<Real>;
    std::vector<Triplet> triplets;

    for (Eigen::Index i = 0; i < t; ++i) {
        for (Eigen::Index j = 0; j < t; ++j) {
            if (dense(i, j) != 0.0) {
                triplets.emplace_back(i, j, dense(i, j));
            }
        }
    }

    SparseMatrixXr sparse(t, t);
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    return sparse;
}

// ============================================================================
// Lag / Recurrence Conversion
// ============================================================================

ArrayXXr recurrence_to_lag(const ArrayXXr& rec, bool pad, int axis) {
    if (axis == -1) axis = 1;

    if (rec.rows() != rec.cols()) {
        throw ParameterError("Non-square recurrence matrix");
    }

    Eigen::Index t = rec.rows();

    ArrayXXr working = rec;

    if (pad) {
        // Pad the non-time axis by t zeros
        // Python: padding[(1 - axis), :] = [0, t]
        if (axis == 1) {
            // Time axis is cols; pad the rows (non-time axis) at the bottom
            ArrayXXr padded = ArrayXXr::Zero(2 * t, t);
            padded.topRows(t) = working;
            working = padded;
        } else {
            // Time axis is rows; pad the cols (non-time axis) at the right
            ArrayXXr padded = ArrayXXr::Zero(t, 2 * t);
            padded.leftCols(t) = working;
            working = padded;
        }
    }

    // Apply shear with factor=-1
    ArrayXXr lag = util::shear(working, -1, axis);
    return lag;
}

ArrayXXr lag_to_recurrence(const ArrayXXr& lag, int axis) {
    if (axis == -1) axis = 1;

    // The time axis has size t; the non-time axis may be 2*t (if padded) or t
    Eigen::Index t;
    if (axis == 1) {
        // Time axis = cols
        t = lag.cols();
    } else {
        // Time axis = rows
        t = lag.rows();
    }

    // Check shape validity
    bool valid = false;
    if (lag.rows() == lag.cols()) {
        valid = true;  // Square (no padding case)
    } else if (axis == 1 && lag.rows() == 2 * t) {
        valid = true;  // Padded on non-time axis (rows)
    } else if (axis == 0 && lag.cols() == 2 * t) {
        valid = true;  // Padded on non-time axis (cols)
    }

    if (!valid) {
        throw ParameterError("Invalid lag matrix shape");
    }

    // Apply inverse shear with factor=+1
    ArrayXXr rec = util::shear(lag, +1, axis);

    // Trim to square (t x t): take only the first t along the non-time axis
    if (axis == 1) {
        // Non-time axis = rows
        return rec.topRows(t);
    } else {
        // Non-time axis = cols
        return rec.leftCols(t);
    }
}

// ============================================================================
// Path Enhancement
// ============================================================================

ArrayXXr path_enhance(
    const ArrayXXr& R,
    int n,
    WindowType window,
    Real max_ratio,
    Real min_ratio,
    int n_filters,
    bool zero_mean,
    bool clip) {

    if (min_ratio <= 0.0) {
        min_ratio = 1.0 / max_ratio;
    }

    if (min_ratio > max_ratio) {
        throw ParameterError("min_ratio cannot exceed max_ratio");
    }

    if (n_filters < 1) {
        throw ParameterError("n_filters must be >= 1");
    }

    ArrayXXr R_smooth;
    bool first = true;

    // Generate logarithmically-spaced ratios
    Real log_min = std::log2(min_ratio);
    Real log_max = std::log2(max_ratio);

    for (int fi = 0; fi < n_filters; ++fi) {
        Real log_ratio;
        if (n_filters == 1) {
            log_ratio = (log_min + log_max) / 2.0;
        } else {
            log_ratio = log_min + fi * (log_max - log_min) / (n_filters - 1);
        }
        Real ratio = std::pow(2.0, log_ratio);

        ArrayXXr kernel = filters::diagonal_filter(window, n, ratio, std::nullopt, zero_mean);

        ArrayXXr convolved = internal::convolve2d(R, kernel, "reflect");

        if (first) {
            R_smooth = convolved;
            first = false;
        } else {
            R_smooth = R_smooth.max(convolved);
        }
    }

    if (clip) {
        R_smooth = R_smooth.max(0.0);
    }

    return R_smooth;
}

// ============================================================================
// Temporal Clustering
// ============================================================================

std::vector<Eigen::Index> agglomerative(
    const ArrayXXr& data,
    int k,
    int axis) {

    if (axis == -1) axis = static_cast<int>(data.rows() > 1 ? data.cols() : data.size()) > 0 ? 1 : 0;
    if (axis < 0) axis += 2;

    // data: (d, n) — features x time
    // Transpose so rows = samples (time steps), cols = features
    ArrayXXr data_T;
    if (axis == 0) {
        data_T = data;  // Already rows = samples
    } else {
        data_T = data.matrix().transpose().array();
    }

    return internal::ward_cluster_1d(data_T, k);
}

std::vector<Eigen::Index> subsegment(
    const ArrayXXr& data,
    const std::vector<Eigen::Index>& frames,
    int n_segments,
    int axis) {

    if (axis == -1) axis = 1;

    if (n_segments < 1) {
        throw ParameterError("n_segments must be a positive integer");
    }

    // Fix frames
    Eigen::Index n = (axis == 0) ? data.rows() : data.cols();
    std::vector<Eigen::Index> fixed_frames = util::fix_frames(frames, 0, n, true);

    std::vector<Eigen::Index> boundaries;

    for (size_t seg = 0; seg + 1 < fixed_frames.size(); ++seg) {
        Eigen::Index seg_start = fixed_frames[seg];
        Eigen::Index seg_end = fixed_frames[seg + 1];
        Eigen::Index seg_len = seg_end - seg_start;

        if (seg_len <= 0) continue;

        int n_seg = std::min(n_segments, static_cast<int>(seg_len));

        // Extract segment data
        ArrayXXr seg_data;
        if (axis == 0) {
            seg_data = data.middleRows(seg_start, seg_len);
        } else {
            seg_data = data.middleCols(seg_start, seg_len);
        }

        // Cluster within segment
        auto sub_bounds = agglomerative(seg_data, n_seg, axis);

        // Offset by segment start
        for (auto b : sub_bounds) {
            boundaries.push_back(seg_start + b);
        }
    }

    // Sort and deduplicate
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

    return boundaries;
}

} // namespace segment
} // namespace librosa
