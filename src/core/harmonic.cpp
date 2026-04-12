#include <librosa/core/harmonic.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include <algorithm>
#include <cmath>

namespace librosa {
namespace core {

namespace {

/// Linear interpolation of y-values at query points xq, given sorted x-values
/// Out-of-bounds queries get fill_value
void linear_interp_1d(const Real* x, const Real* y, Eigen::Index n,
                      const Real* xq, Real* yq, Eigen::Index nq,
                      Real fill_value) {
    for (Eigen::Index i = 0; i < nq; ++i) {
        Real q = xq[i];

        // Out of bounds check
        if (q < x[0] || q > x[n - 1]) {
            yq[i] = fill_value;
            continue;
        }

        // Binary search for the bracketing interval
        auto it = std::lower_bound(x, x + n, q);
        Eigen::Index idx = it - x;

        if (idx == 0) {
            yq[i] = y[0];
        } else if (idx >= n) {
            yq[i] = y[n - 1];
        } else {
            // Linear blend between x[idx-1] and x[idx]
            Real x0 = x[idx - 1];
            Real x1 = x[idx];
            Real t = (x1 > x0) ? (q - x0) / (x1 - x0) : 0.0;
            yq[i] = y[idx - 1] * (1.0 - t) + y[idx] * t;
        }
    }
}

} // anonymous namespace

ArrayXXr interp_harmonics(const ArrayXXr& x, const ArrayXr& freqs,
                          const std::vector<Real>& harmonics,
                          Real fill_value) {
    if (x.rows() != freqs.size()) {
        throw ParameterError("x.rows() must match freqs.size()");
    }
    if (harmonics.empty()) {
        throw ParameterError("harmonics must not be empty");
    }

    Eigen::Index n_freq = x.rows();
    Eigen::Index n_frames = x.cols();
    Eigen::Index n_harmonics = static_cast<Eigen::Index>(harmonics.size());

    ArrayXXr result(n_harmonics * n_freq, n_frames);

    // For each harmonic
    for (Eigen::Index h = 0; h < n_harmonics; ++h) {
        // Compute target frequencies for this harmonic
        ArrayXr target_freqs = freqs * harmonics[h];

        // For each frame, interpolate
        // Note: ArrayXXr is RowMajor, so .col(t) is not contiguous in memory.
        // We must copy columns to/from temporaries for linear_interp_1d.
        for (Eigen::Index t = 0; t < n_frames; ++t) {
            ArrayXr col_in = x.col(t);
            ArrayXr col_out(n_freq);
            linear_interp_1d(
                freqs.data(), col_in.data(), n_freq,
                target_freqs.data(), col_out.data(), n_freq,
                fill_value);
            result.block(h * n_freq, t, n_freq, 1) = col_out;
        }
    }

    return result;
}

ArrayXXr interp_harmonics(const ArrayXr& x, const ArrayXr& freqs,
                          const std::vector<Real>& harmonics,
                          Real fill_value) {
    if (x.size() != freqs.size()) {
        throw ParameterError("x.size() must match freqs.size()");
    }
    if (harmonics.empty()) {
        throw ParameterError("harmonics must not be empty");
    }

    Eigen::Index n_freq = x.size();
    Eigen::Index n_harmonics = static_cast<Eigen::Index>(harmonics.size());

    ArrayXXr result(n_harmonics, n_freq);

    for (Eigen::Index h = 0; h < n_harmonics; ++h) {
        ArrayXr target_freqs = freqs * harmonics[h];
        linear_interp_1d(
            freqs.data(), x.data(), n_freq,
            target_freqs.data(), result.row(h).data(), n_freq,
            fill_value);
    }

    return result;
}

ArrayXXr f0_harmonics(const ArrayXXr& x, const ArrayXr& f0,
                      const ArrayXr& freqs,
                      const std::vector<Real>& harmonics,
                      Real fill_value) {
    if (x.rows() != freqs.size()) {
        throw ParameterError("x.rows() must match freqs.size()");
    }
    if (x.cols() != f0.size()) {
        throw ParameterError("x.cols() must match f0.size()");
    }
    if (harmonics.empty()) {
        throw ParameterError("harmonics must not be empty");
    }

    Eigen::Index n_freq = x.rows();
    Eigen::Index n_frames = x.cols();
    Eigen::Index n_harmonics = static_cast<Eigen::Index>(harmonics.size());

    ArrayXXr result(n_harmonics, n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        // Copy column since ArrayXXr is RowMajor (columns not contiguous)
        ArrayXr col_in = x.col(t);

        // Compute target frequencies for this frame: harmonics * f0[t]
        for (Eigen::Index h = 0; h < n_harmonics; ++h) {
            Real target = harmonics[h] * f0(t);

            // Interpolate x[:,t] at single target frequency
            Real query[1] = {target};
            Real val[1];
            linear_interp_1d(
                freqs.data(), col_in.data(), n_freq,
                query, val, 1, fill_value);
            result(h, t) = val[0];
        }
    }

    return result;
}

ArrayXXr salience(const ArrayXXr& S, const ArrayXr& freqs,
                  const std::vector<Real>& harmonics,
                  const std::vector<Real>& weights,
                  bool filter_peaks, Real fill_value) {
    if (S.rows() != freqs.size()) {
        throw ParameterError("S.rows() must match freqs.size()");
    }
    if (harmonics.empty()) {
        throw ParameterError("harmonics must not be empty");
    }

    Eigen::Index n_freq = S.rows();
    Eigen::Index n_frames = S.cols();
    Eigen::Index n_harmonics = static_cast<Eigen::Index>(harmonics.size());

    // Determine weights
    std::vector<Real> w = weights;
    if (w.empty()) {
        w.assign(harmonics.size(), 1.0);
    }
    if (static_cast<Eigen::Index>(w.size()) != n_harmonics) {
        throw ParameterError("weights size must match harmonics size");
    }

    // Normalize weights to sum to 1
    Real w_sum = 0;
    for (Real wv : w) w_sum += wv;
    if (w_sum > 0) {
        for (Real& wv : w) wv /= w_sum;
    }

    // Compute harmonic interpolation
    ArrayXXr S_harm = interp_harmonics(S, freqs, harmonics, fill_value);

    // Weighted sum across harmonics
    ArrayXXr sal = ArrayXXr::Zero(n_freq, n_frames);
    for (Eigen::Index h = 0; h < n_harmonics; ++h) {
        sal += w[h] * S_harm.block(h * n_freq, 0, n_freq, n_frames);
    }

    // Filter peaks if requested
    if (filter_peaks) {
        auto peaks = util::localmax(sal, 0);
        for (Eigen::Index i = 0; i < sal.rows(); ++i) {
            for (Eigen::Index j = 0; j < sal.cols(); ++j) {
                if (!peaks(i, j)) {
                    sal(i, j) = 0;
                }
            }
        }
    }

    return sal;
}

} // namespace core
} // namespace librosa
