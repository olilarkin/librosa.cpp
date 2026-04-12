#pragma once

#include <librosa/types.hpp>
#include <librosa/util/exceptions.hpp>
#include <Eigen/SVD>
#include <cmath>
#include <algorithm>

namespace librosa {
namespace internal {

/// Result of NMF decomposition
struct NMFResult {
    ArrayXXr W;                 // Components matrix (n_features, n_components)
    ArrayXXr H;                 // Activations matrix (n_components, n_samples)
    Real reconstruction_error;  // ||V - W*H||_F
    int n_iter;                 // Number of iterations performed
};

/// NNDSVD initialization for NMF (Non-Negative Double SVD)
/// Deterministic initialization based on SVD of the input matrix.
/// @param V Input matrix (n_features, n_samples)
/// @param n_components Number of components
/// @return Pair of (W, H) initial matrices
inline std::pair<ArrayXXr, ArrayXXr> nndsvd_init(
    const ArrayXXr& V,
    int n_components) {

    Eigen::Index m = V.rows();
    Eigen::Index n = V.cols();

    // Compute SVD (use Eigen::MatrixXd as the base type)
    Eigen::MatrixXd Vd = V.matrix().cast<double>();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Vd, Eigen::ComputeThinU | Eigen::ComputeThinV);

    auto U = svd.matrixU();
    auto S = svd.singularValues();
    auto Vt = svd.matrixV().transpose();

    ArrayXXr W = ArrayXXr::Zero(m, n_components);
    ArrayXXr H = ArrayXXr::Zero(n_components, n);

    // First component from leading singular value
    Real s0 = S(0);
    W.col(0) = (std::sqrt(s0) * U.col(0).array().abs()).cast<Real>();
    H.row(0) = (std::sqrt(s0) * Vt.row(0).array().abs()).cast<Real>();

    for (int j = 1; j < n_components; ++j) {
        if (j >= S.size()) {
            // Not enough singular values; leave as zero (will be small random)
            Real avg = std::sqrt(V.mean() / n_components);
            W.col(j).setConstant(avg);
            H.row(j).setConstant(avg);
            continue;
        }

        Real sj = S(j);
        auto u = U.col(j);
        auto v = Vt.row(j);

        // Positive and negative parts
        auto u_pos = u.array().max(0.0);
        auto u_neg = (-u.array()).max(0.0);
        auto v_pos = v.array().max(0.0);
        auto v_neg = (-v.array()).max(0.0);

        double u_pos_norm = u_pos.matrix().norm();
        double u_neg_norm = u_neg.matrix().norm();
        double v_pos_norm = v_pos.matrix().norm();
        double v_neg_norm = v_neg.matrix().norm();

        double mp = u_pos_norm * v_pos_norm;
        double mn = u_neg_norm * v_neg_norm;

        if (mp >= mn) {
            Real scale = std::sqrt(sj * mp);
            Real u_scale = (u_pos_norm > 0) ? scale / u_pos_norm : 0.0;
            Real v_scale = (v_pos_norm > 0) ? scale / v_pos_norm : 0.0;
            W.col(j) = (u_pos * u_scale).cast<Real>();
            H.row(j) = (v_pos * v_scale).cast<Real>();
        } else {
            Real scale = std::sqrt(sj * mn);
            Real u_scale = (u_neg_norm > 0) ? scale / u_neg_norm : 0.0;
            Real v_scale = (v_neg_norm > 0) ? scale / v_neg_norm : 0.0;
            W.col(j) = (u_neg * u_scale).cast<Real>();
            H.row(j) = (v_neg * v_scale).cast<Real>();
        }
    }

    // Replace exact zeros with small values to avoid division issues
    Real eps = std::sqrt(std::numeric_limits<Real>::epsilon());
    Real avg = V.mean();
    Real fill_val = eps * avg;
    if (fill_val == 0.0) fill_val = eps;

    W = W.max(fill_val);
    H = H.max(fill_val);

    return {W, H};
}

/// NMF using multiplicative update rules (Frobenius norm)
/// Solves: min_{W>=0, H>=0} ||V - W*H||_F^2
/// @param V Input non-negative matrix (n_features, n_samples)
/// @param n_components Number of components
/// @param max_iter Maximum number of iterations
/// @param tol Convergence tolerance (relative change in error)
/// @return NMFResult with W, H, reconstruction error, and iteration count
inline NMFResult nmf_mu(
    const ArrayXXr& V,
    int n_components,
    int max_iter = 200,
    Real tol = 1e-4) {

    if (n_components <= 0) {
        throw ParameterError("n_components must be positive");
    }
    if ((V < 0).any()) {
        throw ParameterError("Input matrix must be non-negative for NMF");
    }

    Eigen::Index m = V.rows();    // n_features
    Eigen::Index n = V.cols();    // n_samples

    if (n_components > std::min(m, n)) {
        n_components = static_cast<int>(std::min(m, n));
    }

    // NNDSVD initialization
    auto [W, H] = nndsvd_init(V, n_components);

    // Convert to matrices for efficient multiplication
    MatrixXr Wm = W.matrix();
    MatrixXr Hm = H.matrix();
    MatrixXr Vm = V.matrix();

    Real eps = std::numeric_limits<Real>::epsilon();

    // Initial reconstruction error
    MatrixXr WH = Wm * Hm;
    Real prev_error = (Vm - WH).norm();
    Real init_error = prev_error;

    int n_iter = 0;
    for (int iter = 0; iter < max_iter; ++iter) {
        n_iter = iter + 1;

        // Update H: H *= (W^T * V) / (W^T * W * H + eps)
        MatrixXr WtV = Wm.transpose() * Vm;
        MatrixXr WtWH = Wm.transpose() * Wm * Hm;
        Hm.array() *= WtV.array() / (WtWH.array() + eps);
        Hm = Hm.array().max(eps).matrix();

        // Update W: W *= (V * H^T) / (W * H * H^T + eps)
        MatrixXr VHt = Vm * Hm.transpose();
        MatrixXr WHHt = Wm * Hm * Hm.transpose();
        Wm.array() *= VHt.array() / (WHHt.array() + eps).matrix().array();
        Wm = Wm.array().max(eps).matrix();

        // Check convergence every 10 iterations
        if ((iter + 1) % 10 == 0 || iter == max_iter - 1) {
            WH = Wm * Hm;
            Real error = (Vm - WH).norm();

            if (init_error > 0 && std::abs(prev_error - error) / init_error < tol) {
                prev_error = error;
                break;
            }
            prev_error = error;
        }
    }

    return NMFResult{
        Wm.array(),
        Hm.array(),
        prev_error,
        n_iter
    };
}

} // namespace internal
} // namespace librosa
