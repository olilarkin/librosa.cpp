#pragma once

/// Internal DCT-II and DCT-III implementations shared across feature modules.
/// Not part of the public API.

#include <librosa/types.hpp>
#include <cmath>

namespace librosa {
namespace internal {

/// DCT-II implementation
/// Equivalent to scipy.fft.dct(x_padded, type=2) where x is zero-padded to max(n_freq, n_out).
/// @param X Input matrix [shape: (n_freq, n_frames)]
/// @param n_out Number of output coefficients
/// @param ortho Apply orthonormal normalization
/// @return DCT-II coefficients [shape: (n_out, n_frames)]
inline ArrayXXr dct_ii(const ArrayXXr& X, int n_out, bool ortho = true) {
    Eigen::Index n_freq = X.rows();
    Eigen::Index n_frames = X.cols();
    // Transform length N: when expanding (n_out > n_freq), input is zero-padded to n_out
    Eigen::Index N = std::max(n_freq, static_cast<Eigen::Index>(n_out));

    ArrayXXr result(n_out, n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        for (int k = 0; k < n_out; ++k) {
            Real sum = 0;
            // Only iterate over non-zero inputs (up to n_freq)
            for (Eigen::Index n = 0; n < n_freq; ++n) {
                sum += X(n, t) * std::cos(M_PI * k * (2.0 * n + 1) / (2.0 * N));
            }
            result(k, t) = sum;
        }
    }

    // Orthonormal normalization uses transform length N
    if (ortho) {
        Real scale0 = std::sqrt(1.0 / N);
        Real scale = std::sqrt(2.0 / N);
        result.row(0) *= scale0;
        for (int k = 1; k < n_out; ++k) {
            result.row(k) *= scale;
        }
    }

    return result;
}

/// DCT-III implementation (inverse of DCT-II)
/// Equivalent to scipy.fft.idct(x, type=2, n=n_out) or scipy.fft.dct(x_padded, type=3)
/// When n_out > n_freq, the input is implicitly zero-padded to n_out before transform.
/// @param X Input matrix [shape: (n_freq, n_frames)]
/// @param n_out Number of output coefficients (transform length N)
/// @param ortho Apply orthonormal normalization
/// @return DCT-III coefficients [shape: (n_out, n_frames)]
inline ArrayXXr dct_iii(const ArrayXXr& X, int n_out, bool ortho = true) {
    Eigen::Index n_freq = X.rows();
    Eigen::Index n_frames = X.cols();
    // Transform length N = n_out (input is zero-padded to this length)
    Eigen::Index N = n_out;

    ArrayXXr result(n_out, n_frames);

    if (ortho) {
        // Orthonormal DCT-III:
        // y[k] = x[0]/sqrt(N) + sqrt(2/N) * sum_{n=1}^{N-1} x[n]*cos(pi*n*(2k+1)/(2N))
        Real scale0 = 1.0 / std::sqrt(static_cast<Real>(N));
        Real scale = std::sqrt(2.0 / N);
        Eigen::Index n_max = std::min(n_freq, N);
        for (Eigen::Index t = 0; t < n_frames; ++t) {
            for (int k = 0; k < n_out; ++k) {
                Real sum = X(0, t) * scale0;
                for (Eigen::Index n = 1; n < n_max; ++n) {
                    sum += scale * X(n, t) * std::cos(M_PI * n * (2.0 * k + 1) / (2.0 * N));
                }
                result(k, t) = sum;
            }
        }
    } else {
        Eigen::Index n_max = std::min(n_freq, N);
        for (Eigen::Index t = 0; t < n_frames; ++t) {
            for (int k = 0; k < n_out; ++k) {
                Real sum = X(0, t) / 2;
                for (Eigen::Index n = 1; n < n_max; ++n) {
                    sum += X(n, t) * std::cos(M_PI * n * (2.0 * k + 1) / (2.0 * N));
                }
                result(k, t) = sum * 2;
            }
        }
    }

    return result;
}

} // namespace internal
} // namespace librosa
