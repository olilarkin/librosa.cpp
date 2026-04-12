#include <librosa/decompose.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include "internal/nmf.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>

namespace librosa {
namespace decompose {

// ============================================================================
// Median Filter Implementation
// ============================================================================

namespace {

// Pad a 2D array with specified mode
ArrayXXr pad_array_2d(
    const ArrayXXr& S,
    int pad_rows,
    int pad_cols,
    const std::string& mode) {

    Eigen::Index rows = S.rows();
    Eigen::Index cols = S.cols();
    Eigen::Index new_rows = rows + 2 * pad_rows;
    Eigen::Index new_cols = cols + 2 * pad_cols;

    ArrayXXr padded = ArrayXXr::Zero(new_rows, new_cols);

    // Copy center
    padded.block(pad_rows, pad_cols, rows, cols) = S;

    if (mode == "constant") {
        // Already zero-padded
    } else if (mode == "edge") {
        // Top edge
        for (int i = 0; i < pad_rows; ++i) {
            padded.row(i).segment(pad_cols, cols) = S.row(0);
        }
        // Bottom edge
        for (int i = 0; i < pad_rows; ++i) {
            padded.row(new_rows - 1 - i).segment(pad_cols, cols) = S.row(rows - 1);
        }
        // Left edge
        for (int j = 0; j < pad_cols; ++j) {
            padded.col(j).segment(pad_rows, rows) = S.col(0);
        }
        // Right edge
        for (int j = 0; j < pad_cols; ++j) {
            padded.col(new_cols - 1 - j).segment(pad_rows, rows) = S.col(cols - 1);
        }
        // Corners
        for (int i = 0; i < pad_rows; ++i) {
            for (int j = 0; j < pad_cols; ++j) {
                padded(i, j) = S(0, 0);
                padded(i, new_cols - 1 - j) = S(0, cols - 1);
                padded(new_rows - 1 - i, j) = S(rows - 1, 0);
                padded(new_rows - 1 - i, new_cols - 1 - j) = S(rows - 1, cols - 1);
            }
        }
    } else if (mode == "reflect") {
        // Top edge (reflect without edge)
        for (int i = 0; i < pad_rows; ++i) {
            int src_row = std::min(i + 1, static_cast<int>(rows - 1));
            padded.row(pad_rows - 1 - i).segment(pad_cols, cols) = S.row(src_row);
        }
        // Bottom edge
        for (int i = 0; i < pad_rows; ++i) {
            int src_row = std::max(0, static_cast<int>(rows - 2 - i));
            padded.row(pad_rows + rows + i).segment(pad_cols, cols) = S.row(src_row);
        }
        // Left edge
        for (int j = 0; j < pad_cols; ++j) {
            int src_col = std::min(j + 1, static_cast<int>(cols - 1));
            padded.col(pad_cols - 1 - j).segment(pad_rows, rows) = S.col(src_col);
        }
        // Right edge
        for (int j = 0; j < pad_cols; ++j) {
            int src_col = std::max(0, static_cast<int>(cols - 2 - j));
            padded.col(pad_cols + cols + j).segment(pad_rows, rows) = S.col(src_col);
        }
        // Corners - reflect in both directions
        for (int i = 0; i < pad_rows; ++i) {
            for (int j = 0; j < pad_cols; ++j) {
                int src_row_top = std::min(i + 1, static_cast<int>(rows - 1));
                int src_row_bot = std::max(0, static_cast<int>(rows - 2 - i));
                int src_col_left = std::min(j + 1, static_cast<int>(cols - 1));
                int src_col_right = std::max(0, static_cast<int>(cols - 2 - j));

                padded(pad_rows - 1 - i, pad_cols - 1 - j) = S(src_row_top, src_col_left);
                padded(pad_rows - 1 - i, pad_cols + cols + j) = S(src_row_top, src_col_right);
                padded(pad_rows + rows + i, pad_cols - 1 - j) = S(src_row_bot, src_col_left);
                padded(pad_rows + rows + i, pad_cols + cols + j) = S(src_row_bot, src_col_right);
            }
        }
    } else {
        throw ParameterError("Unknown padding mode: " + mode);
    }

    return padded;
}

// Compute median of a vector
Real compute_median(std::vector<Real>& values) {
    if (values.empty()) return 0.0;

    size_t n = values.size();
    size_t mid = n / 2;

    std::nth_element(values.begin(), values.begin() + mid, values.end());

    if (n % 2 == 0) {
        Real median_high = values[mid];
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        Real median_low = values[mid - 1];
        return (median_low + median_high) / 2.0;
    }

    return values[mid];
}

} // anonymous namespace

ArrayXXr median_filter_2d(
    const ArrayXXr& S,
    std::pair<int, int> size,
    const std::string& mode) {

    int filter_rows = size.first;
    int filter_cols = size.second;

    if (filter_rows < 1 || filter_cols < 1) {
        throw ParameterError("Filter size must be positive");
    }

    // For odd filter sizes, pad by half the size
    int pad_rows = filter_rows / 2;
    int pad_cols = filter_cols / 2;

    // Pad the input
    ArrayXXr padded = pad_array_2d(S, pad_rows, pad_cols, mode);

    // Output array
    ArrayXXr result(S.rows(), S.cols());

    // Buffer for computing median
    std::vector<Real> window(filter_rows * filter_cols);

    // Apply median filter
    for (Eigen::Index i = 0; i < S.rows(); ++i) {
        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            // Extract window
            int idx = 0;
            for (int fi = 0; fi < filter_rows; ++fi) {
                for (int fj = 0; fj < filter_cols; ++fj) {
                    window[idx++] = padded(i + fi, j + fj);
                }
            }

            result(i, j) = compute_median(window);
        }
    }

    return result;
}

// ============================================================================
// HPSS Implementation
// ============================================================================

namespace {

std::pair<ArrayXXr, ArrayXXr> hpss_impl(
    const ArrayXXr& S,
    std::pair<int, int> kernel_size,
    Real power,
    bool mask,
    std::pair<Real, Real> margin) {

    int win_harm = kernel_size.first;
    int win_perc = kernel_size.second;
    Real margin_harm = margin.first;
    Real margin_perc = margin.second;

    if (margin_harm < 1.0 || margin_perc < 1.0) {
        throw ParameterError("Margins must be >= 1.0. A typical range is between 1 and 10.");
    }

    // Apply median filters
    // Harmonic filter: along time axis (columns), so shape is (1, win_harm)
    // Percussive filter: along frequency axis (rows), so shape is (win_perc, 1)
    ArrayXXr harm = median_filter_2d(S, {1, win_harm}, "reflect");
    ArrayXXr perc = median_filter_2d(S, {win_perc, 1}, "reflect");

    // Compute soft masks
    bool split_zeros = (margin_harm == 1.0 && margin_perc == 1.0);

    ArrayXXr mask_harm = util::softmask(harm, perc * margin_harm, power, split_zeros);
    ArrayXXr mask_perc = util::softmask(perc, harm * margin_perc, power, split_zeros);

    if (mask) {
        return {mask_harm, mask_perc};
    }

    return {S * mask_harm, S * mask_perc};
}

} // anonymous namespace

// Single kernel size, single margin
std::pair<ArrayXXr, ArrayXXr> hpss(
    const ArrayXXr& S,
    int kernel_size,
    Real power,
    bool mask,
    Real margin) {

    return hpss_impl(S, {kernel_size, kernel_size}, power, mask, {margin, margin});
}

// Separate kernel sizes, single margin
std::pair<ArrayXXr, ArrayXXr> hpss(
    const ArrayXXr& S,
    std::pair<int, int> kernel_size,
    Real power,
    bool mask,
    Real margin) {

    return hpss_impl(S, kernel_size, power, mask, {margin, margin});
}

// Single kernel size, separate margins
std::pair<ArrayXXr, ArrayXXr> hpss(
    const ArrayXXr& S,
    int kernel_size,
    Real power,
    bool mask,
    std::pair<Real, Real> margin) {

    return hpss_impl(S, {kernel_size, kernel_size}, power, mask, margin);
}

// Separate kernel sizes and margins
std::pair<ArrayXXr, ArrayXXr> hpss(
    const ArrayXXr& S,
    std::pair<int, int> kernel_size,
    Real power,
    bool mask,
    std::pair<Real, Real> margin) {

    return hpss_impl(S, kernel_size, power, mask, margin);
}

// ============================================================================
// Complex HPSS Implementation
// ============================================================================

namespace {

std::pair<ArrayXXc, ArrayXXc> hpss_complex_impl(
    const ArrayXXc& S,
    std::pair<int, int> kernel_size,
    Real power,
    bool mask,
    std::pair<Real, Real> margin) {

    // Extract magnitude and phase
    ArrayXXr mag = S.abs();
    ArrayXXc phase = S / (mag.max(1e-10).cast<Complex>());

    // Apply HPSS on magnitude
    auto [harm_mag, perc_mag] = hpss_impl(mag, kernel_size, power, mask, margin);

    if (mask) {
        // Return real masks cast to complex
        ArrayXXc harm_mask = harm_mag.cast<Complex>();
        ArrayXXc perc_mask = perc_mag.cast<Complex>();
        return {harm_mask, perc_mask};
    }

    // Reconstruct with phase
    ArrayXXc harm = harm_mag.cast<Complex>() * phase;
    ArrayXXc perc = perc_mag.cast<Complex>() * phase;

    return {harm, perc};
}

} // anonymous namespace

// Single kernel size, single margin
std::pair<ArrayXXc, ArrayXXc> hpss_complex(
    const ArrayXXc& S,
    int kernel_size,
    Real power,
    bool mask,
    Real margin) {

    return hpss_complex_impl(S, {kernel_size, kernel_size}, power, mask, {margin, margin});
}

// Separate kernel sizes, single margin
std::pair<ArrayXXc, ArrayXXc> hpss_complex(
    const ArrayXXc& S,
    std::pair<int, int> kernel_size,
    Real power,
    bool mask,
    Real margin) {

    return hpss_complex_impl(S, kernel_size, power, mask, {margin, margin});
}

// Single kernel size, separate margins
std::pair<ArrayXXc, ArrayXXc> hpss_complex(
    const ArrayXXc& S,
    int kernel_size,
    Real power,
    bool mask,
    std::pair<Real, Real> margin) {

    return hpss_complex_impl(S, {kernel_size, kernel_size}, power, mask, margin);
}

// Separate kernel sizes and margins
std::pair<ArrayXXc, ArrayXXc> hpss_complex(
    const ArrayXXc& S,
    std::pair<int, int> kernel_size,
    Real power,
    bool mask,
    std::pair<Real, Real> margin) {

    return hpss_complex_impl(S, kernel_size, power, mask, margin);
}

// ============================================================================
// NMF Decomposition
// ============================================================================

std::pair<ArrayXXr, ArrayXXr> decompose_nmf(
    const ArrayXXr& S,
    int n_components,
    bool sort,
    int max_iter,
    Real tol) {

    Eigen::Index n_features = S.rows();
    Eigen::Index n_samples = S.cols();

    if (n_components <= 0) {
        n_components = static_cast<int>(n_features);
    }

    // NMF convention: V = W * H
    // librosa convention: S (n_features x n_samples)
    // sklearn NMF: fit_transform(S.T) where S.T is (n_samples x n_features)
    //   transformer.components_ is (n_components x n_features)
    //   activations = fit_transform(S.T).T is (n_components x n_samples)
    //   components = transformer.components_.T is (n_features x n_components)

    // Our internal NMF: V = W*H where V is (n_features x n_samples)
    //   W is (n_features x n_components) = components
    //   H is (n_components x n_samples) = activations

    auto result = internal::nmf_mu(S, n_components, max_iter, tol);

    ArrayXXr components = result.W;    // (n_features, n_components)
    ArrayXXr activations = result.H;   // (n_components, n_samples)

    if (sort) {
        auto [sorted_components, indices] = util::axis_sort_with_index(components, -1);
        components = sorted_components;

        // Reorder activations to match
        ArrayXXr sorted_activations(activations.rows(), activations.cols());
        for (size_t i = 0; i < indices.size(); ++i) {
            sorted_activations.row(static_cast<Eigen::Index>(i)) =
                activations.row(indices[i]);
        }
        activations = sorted_activations;
    }

    return {components, activations};
}

// ============================================================================
// Nearest-Neighbor Filter
// ============================================================================

ArrayXXr nn_filter(
    const ArrayXXr& S,
    const SparseMatrixXr& rec,
    AggregateFunc aggregate,
    int axis) {

    if (axis == -1) axis = 1;

    Eigen::Index n_obs = (axis == 0) ? S.rows() : S.cols();

    if (rec.rows() != n_obs || rec.cols() != n_obs) {
        throw ParameterError(
            "Invalid recurrence matrix shape for S");
    }

    // Work with observations on axis 0
    ArrayXXr S_work;
    if (axis == 0) {
        S_work = S;
    } else {
        S_work = S.matrix().transpose().array();
    }

    ArrayXXr S_out(S_work.rows(), S_work.cols());

    for (Eigen::Index i = 0; i < n_obs; ++i) {
        // Get neighbors from recurrence matrix row i
        std::vector<Eigen::Index> neighbors;
        std::vector<Real> weights;

        for (SparseMatrixXr::InnerIterator it(rec, i); it; ++it) {
            neighbors.push_back(it.col());
            weights.push_back(it.value());
        }

        if (neighbors.empty()) {
            S_out.row(i) = S_work.row(i);
            continue;
        }

        Eigen::Index n_neighbors = static_cast<Eigen::Index>(neighbors.size());
        Eigen::Index n_features = S_work.cols();

        if (aggregate == AggregateFunc::Mean) {
            // Weighted mean using recurrence weights
            ArrayXr result = ArrayXr::Zero(n_features);
            Real total_weight = 0.0;
            for (size_t ni = 0; ni < neighbors.size(); ++ni) {
                result += weights[ni] * S_work.row(neighbors[ni]).transpose();
                total_weight += weights[ni];
            }
            if (total_weight > 0) {
                S_out.row(i) = (result / total_weight).transpose();
            } else {
                S_out.row(i) = S_work.row(i);
            }
        } else if (aggregate == AggregateFunc::Median) {
            // Median across neighbors for each feature
            std::vector<Real> vals(n_neighbors);
            for (Eigen::Index f = 0; f < n_features; ++f) {
                for (Eigen::Index ni = 0; ni < n_neighbors; ++ni) {
                    vals[ni] = S_work(neighbors[ni], f);
                }
                std::sort(vals.begin(), vals.end());
                if (n_neighbors % 2 == 0) {
                    S_out(i, f) = (vals[n_neighbors / 2 - 1] + vals[n_neighbors / 2]) / 2.0;
                } else {
                    S_out(i, f) = vals[n_neighbors / 2];
                }
            }
        } else if (aggregate == AggregateFunc::Min) {
            ArrayXr result = ArrayXr::Constant(n_features, std::numeric_limits<Real>::infinity());
            for (auto ni : neighbors) {
                result = result.min(S_work.row(ni).transpose());
            }
            S_out.row(i) = result.transpose();
        } else if (aggregate == AggregateFunc::Max) {
            ArrayXr result = ArrayXr::Constant(n_features, -std::numeric_limits<Real>::infinity());
            for (auto ni : neighbors) {
                result = result.max(S_work.row(ni).transpose());
            }
            S_out.row(i) = result.transpose();
        }
    }

    // Transpose back if needed
    if (axis != 0) {
        return S_out.matrix().transpose().array();
    }
    return S_out;
}

} // namespace decompose
} // namespace librosa
