#pragma once

#include "../types.hpp"
#include "exceptions.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace librosa {
namespace util {

/// Maximum memory block size for STFT computations (256 KB)
constexpr size_t MAX_MEM_BLOCK = 256 * 1024;

/// Get the smallest representable positive number for a floating point type
template<typename T>
T tiny(const T& = T()) {
    return std::numeric_limits<T>::min();
}

/// Validate that audio data is valid (finite, floating-point)
/// @param y Audio data to validate
/// @return true if valid
/// @throws ParameterError if invalid
bool valid_audio(const ArrayXr& y);
bool valid_audio(const ArrayXXr& y);

/// Check if x is a positive integer
bool is_positive_int(int x);

/// Validate a value as an integer
/// @param x Value to convert
/// @param use_floor If true, floor the value (default), otherwise use as-is
/// @return Integer representation
int valid_int(double x, bool use_floor = true);

/// Validate interval array
/// @param intervals Array of shape (n, 2) representing time intervals
/// @return true if valid
/// @throws ParameterError if invalid
bool valid_intervals(const ArrayXXr& intervals);

/// Pad an array to a target length, centering the data
/// @param data Input array
/// @param size Target size
/// @param axis Axis along which to pad (default: -1, meaning last axis)
/// @param mode Padding mode
/// @param constant_value Value for constant padding
/// @return Padded array
ArrayXr pad_center(const ArrayXr& data, Eigen::Index size,
                   PadMode mode = PadMode::Constant,
                   Real constant_value = 0.0);

ArrayXXr pad_center(const ArrayXXr& data, Eigen::Index size, int axis = -1,
                    PadMode mode = PadMode::Constant,
                    Real constant_value = 0.0);

/// Fix the length of an array to exactly size
/// @param data Input array
/// @param size Target size
/// @param axis Axis along which to fix length
/// @param mode Padding mode if extending
/// @return Array of exactly the target size
ArrayXr fix_length(const ArrayXr& data, Eigen::Index size,
                   PadMode mode = PadMode::Constant);

ArrayXXr fix_length(const ArrayXXr& data, Eigen::Index size, int axis = -1,
                    PadMode mode = PadMode::Constant);

/// Fix a list of frames to lie within [x_min, x_max]
/// @param frames Frame indices
/// @param x_min Minimum allowed frame index (default: 0)
/// @param x_max Maximum allowed frame index (default: nullopt)
/// @param pad If true, pad to span full range
/// @return Fixed frame indices, sorted and unique
std::vector<Eigen::Index> fix_frames(const std::vector<Eigen::Index>& frames,
                                      Eigen::Index x_min = 0,
                                      std::optional<Eigen::Index> x_max = std::nullopt,
                                      bool pad = true);

/// Frame an array into overlapping windows
/// @param x Input array
/// @param frame_length Length of each frame
/// @param hop_length Number of samples between frames
/// @return Framed array of shape (frame_length, n_frames)
ArrayXXr frame(const ArrayXr& x, Eigen::Index frame_length, Eigen::Index hop_length);

/// Normalize an array along a chosen axis
/// @param S Input array
/// @param norm Normalization type (inf, -inf, 0, or p-norm)
/// @param axis Axis along which to normalize
/// @param threshold Minimum norm threshold
/// @param fill How to handle small-norm slices (nullopt=leave, false=zero, true=fill)
/// @return Normalized array
ArrayXXr normalize(const ArrayXXr& S, Real norm = std::numeric_limits<Real>::infinity(),
                   int axis = 0, std::optional<Real> threshold = std::nullopt,
                   std::optional<bool> fill = std::nullopt);

ArrayXr normalize(const ArrayXr& S, Real norm = std::numeric_limits<Real>::infinity(),
                  std::optional<Real> threshold = std::nullopt,
                  std::optional<bool> fill = std::nullopt);

/// Find local maxima in an array
/// @param x Input array
/// @param axis Axis along which to find maxima
/// @return Boolean array indicating local maxima
Eigen::Array<bool, Eigen::Dynamic, 1> localmax(const ArrayXr& x);
Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
localmax(const ArrayXXr& x, int axis = 0);

/// Find local minima in an array
/// @param x Input array
/// @param axis Axis along which to find minima
/// @return Boolean array indicating local minima
Eigen::Array<bool, Eigen::Dynamic, 1> localmin(const ArrayXr& x);
Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
localmin(const ArrayXXr& x, int axis = 0);

/// Pick peaks in a signal using flexible heuristics
/// @param x Input signal
/// @param pre_max Samples before n for max computation
/// @param post_max Samples after n for max computation
/// @param pre_avg Samples before n for mean computation
/// @param post_avg Samples after n for mean computation
/// @param delta Threshold offset for mean
/// @param wait Samples to wait after picking a peak
/// @return Indices of peaks
std::vector<Eigen::Index> peak_pick(const ArrayXr& x,
                                    int pre_max, int post_max,
                                    int pre_avg, int post_avg,
                                    Real delta, int wait);

/// Sort a 2D array along rows or columns
/// @param S Input array
/// @param axis Axis along which to sort
/// @return Sorted array
ArrayXXr axis_sort(const ArrayXXr& S, int axis = -1);

/// Sort a 2D array and return indices
/// @param S Input array
/// @param axis Axis along which to sort
/// @return Pair of (sorted array, indices)
std::pair<ArrayXXr, std::vector<Eigen::Index>>
axis_sort_with_index(const ArrayXXr& S, int axis = -1);

/// Sparsify rows of an array
/// @param x Input array
/// @param quantile Quantile below which to zero out
/// @return Sparsified array
ArrayXXr sparsify_rows(const ArrayXXr& x, Real quantile = 0.01);

/// Compute softmask between two arrays
/// @param X First array
/// @param X_ref Reference array
/// @param power Exponent for the mask
/// @param split_zeros How to handle zeros
/// @return Softmask array
ArrayXXr softmask(const ArrayXXr& X, const ArrayXXr& X_ref,
                  Real power = 1.0, bool split_zeros = false);

/// Synchronize aggregated feature values between frames
/// @param data Feature matrix
/// @param idx Frame indices to synchronize on
/// @param aggregate Aggregation function
/// @param pad Whether to pad the output
/// @param axis Axis to aggregate along
/// @return Synchronized features
ArrayXXr sync(const ArrayXXr& data, const std::vector<Eigen::Index>& idx,
              AggregateFunc aggregate = AggregateFunc::Mean,
              bool pad = true, int axis = -1);

/// Compute absolute value squared (more efficient than abs then square)
/// @param x Complex input array
/// @return |x|^2
ArrayXr abs2(const ArrayXc& x);
ArrayXXr abs2(const ArrayXXc& x);

/// Create a phasor (complex exponential) from phase values
/// @param angles Phase angles in radians
/// @param mag Optional magnitude (default 1)
/// @return Complex phasor e^(i*angles) * mag
ArrayXc phasor(const ArrayXr& angles, const ArrayXr& mag);
ArrayXc phasor(const ArrayXr& angles, Real mag = 1.0);

/// Compute the cyclic gradient of a phase array
/// @param x Phase values
/// @param edge_order Order of gradient at edges
/// @return Cyclic gradient
ArrayXr cyclic_gradient(const ArrayXr& x, int edge_order = 1);

/// Get complex dtype from real dtype
/// For float -> complex<float>, double -> complex<double>
template<typename T>
using complex_of = std::complex<T>;

/// Fill off-diagonal elements of a square matrix
/// @param x Square matrix to fill
/// @param fill_value Value to fill with
/// @param wrap Whether to wrap around
/// @return Filled matrix
ArrayXXr fill_off_diagonal(const ArrayXXr& x, Real fill_value, bool wrap = false);

/// Stack arrays with optional padding
/// @param arrays Input arrays
/// @param axis Axis for stacking
/// @return Stacked array
ArrayXXr stack(const std::vector<ArrayXr>& arrays, int axis = 0);

/// Convert buffer to float array
/// @param buf Input buffer (int16, int32, etc.)
/// @param n_bytes Bytes per sample
/// @return Normalized float array [-1, 1]
ArrayXr buf_to_float(const void* buf, size_t n_samples, int n_bytes);

/// Check if all elements in array are unique
bool is_unique(const ArrayXr& x);

/// Count unique elements
size_t count_unique(const ArrayXr& x);

/// Expand dimensions of array
/// @param x Input array
/// @param ndim Target number of dimensions
/// @param axes Target axes for existing dimensions
/// @return Expanded array
// Note: This is complex to implement generically in C++,
// we provide specific overloads instead

/// Shear a matrix along the specified axis
ArrayXXr shear(const ArrayXXr& X, int factor = 1, int axis = -1);

/// Convert slice indices to a vector of indices
std::vector<Eigen::Index> index_to_slice(Eigen::Index idx,
                                          Eigen::Index idx_min = 0,
                                          std::optional<Eigen::Index> idx_max = std::nullopt,
                                          Eigen::Index step = 1,
                                          bool pad = true);

/// Match one set of time intervals to another using Jaccard similarity.
/// Each interval [a, b] from intervals_from is matched to the interval [c, d]
/// from intervals_to which maximizes Jaccard similarity.
/// @param intervals_from Source intervals, shape (n, 2)
/// @param intervals_to Target intervals, shape (m, 2)
/// @param strict If true, throw if no overlapping interval exists
/// @return Vector of indices into intervals_to for each interval in intervals_from
std::vector<Eigen::Index> match_intervals(
    const ArrayXXr& intervals_from,
    const ArrayXXr& intervals_to,
    bool strict = true);

/// Non-negative least squares solver
/// Finds X >= 0 minimizing ||A*X - B||^2, solving column by column
/// @param A Coefficient matrix [shape: (m, n)]
/// @param B Target matrix [shape: (m, k)]
/// @return Solution X [shape: (n, k)]
ArrayXXr nnls(const MatrixXr& A, const ArrayXXr& B);

} // namespace util
} // namespace librosa
