#pragma once

#include <librosa/types.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>
#include <algorithm>

namespace librosa {
namespace internal {

/// Check YIN parameters for validity
inline void check_yin_params(Real sr, Real fmax, Real fmin, int frame_length) {
    if (fmax > sr / 2) {
        throw ParameterError("fmax cannot exceed Nyquist frequency");
    }
    if (fmin >= fmax) {
        throw ParameterError("fmin must be less than fmax");
    }
    if (fmin <= 0) {
        throw ParameterError("fmin must be strictly positive");
    }
    if (sr / fmin >= frame_length - 1) {
        throw ParameterError("fmin is too small for the given frame_length");
    }
}

/// Cumulative mean normalized difference function for YIN
inline ArrayXXr cumulative_mean_normalized_difference(
    const ArrayXXr& y_frames,
    int min_period,
    int max_period) {

    Eigen::Index frame_length = y_frames.rows();
    Eigen::Index n_frames = y_frames.cols();

    // Compute autocorrelation
    ArrayXXr acf_frames = autocorrelate(y_frames, max_period + 1, 0);

    // Energy terms (cumulative sum of squared samples)
    ArrayXXr energy = y_frames.square();
    for (Eigen::Index j = 0; j < n_frames; ++j) {
        for (Eigen::Index i = 1; i < frame_length; ++i) {
            energy(i, j) += energy(i - 1, j);
        }
    }

    // Compute difference function
    ArrayXXr yin_frames = ArrayXXr::Zero(max_period + 1, n_frames);

    for (Eigen::Index j = 0; j < n_frames; ++j) {
        yin_frames(0, j) = 0;
        for (int k = 1; k <= max_period; ++k) {
            yin_frames(k, j) = 2 * (acf_frames(0, j) - acf_frames(k, j)) - energy(k - 1, j);
        }
    }

    // Cumulative mean normalized difference
    ArrayXXr result = ArrayXXr::Zero(max_period - min_period + 1, n_frames);

    for (Eigen::Index j = 0; j < n_frames; ++j) {
        Real cumsum = 0;
        for (int k = 1; k <= max_period; ++k) {
            cumsum += yin_frames(k, j);
            Real cumulative_mean = cumsum / k;

            if (k >= min_period) {
                int idx = k - min_period;
                Real denom = cumulative_mean + util::tiny(cumulative_mean);
                result(idx, j) = yin_frames(k, j) / denom;
            }
        }
    }

    return result;
}

/// Parabolic interpolation for pitch refinement
inline ArrayXXr parabolic_interpolation(const ArrayXXr& x) {
    ArrayXXr shifts = ArrayXXr::Zero(x.rows(), x.cols());

    for (Eigen::Index j = 0; j < x.cols(); ++j) {
        for (Eigen::Index i = 1; i < x.rows() - 1; ++i) {
            Real x_prev = x(i - 1, j);
            Real x_curr = x(i, j);
            Real x_next = x(i + 1, j);

            Real a = x_next + x_prev - 2 * x_curr;
            Real b = (x_next - x_prev) / 2;

            if (std::abs(b) >= std::abs(a)) {
                shifts(i, j) = 0;
            } else if (std::abs(a) > util::tiny(a)) {
                shifts(i, j) = -b / a;
            }
        }
    }

    return shifts;
}

} // namespace internal
} // namespace librosa
