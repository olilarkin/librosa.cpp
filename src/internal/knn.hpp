#pragma once

#include <librosa/types.hpp>
#include <librosa/util/exceptions.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace librosa {
namespace internal {

/// Compute pairwise Euclidean distance between rows of X and Y.
/// @param X Matrix of shape (n, d)
/// @param Y Matrix of shape (m, d)
/// @return Distance matrix of shape (n, m)
inline ArrayXXr cdist_euclidean_rows(const ArrayXXr& X, const ArrayXXr& Y) {
    // ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
    MatrixXr Xm = X.matrix();
    MatrixXr Ym = Y.matrix();

    ArrayXr X_sq = X.square().rowwise().sum();  // (n,)
    ArrayXr Y_sq = Y.square().rowwise().sum();  // (m,)

    // XY = X @ Y^T, shape (n, m)
    MatrixXr XY = Xm * Ym.transpose();

    Eigen::Index n = X.rows();
    Eigen::Index m = Y.rows();
    ArrayXXr dist(n, m);

    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < m; ++j) {
            Real d2 = X_sq(i) + Y_sq(j) - 2.0 * XY(i, j);
            dist(i, j) = std::sqrt(std::max(d2, Real(0.0)));
        }
    }

    return dist;
}

/// Compute pairwise cosine distance between rows of X and Y.
/// cosine_dist(x, y) = 1 - (x.y) / (||x|| * ||y||)
/// @param X Matrix of shape (n, d)
/// @param Y Matrix of shape (m, d)
/// @return Distance matrix of shape (n, m)
inline ArrayXXr cdist_cosine_rows(const ArrayXXr& X, const ArrayXXr& Y) {
    MatrixXr Xm = X.matrix();
    MatrixXr Ym = Y.matrix();

    ArrayXr X_norm = Xm.rowwise().norm().array();
    ArrayXr Y_norm = Ym.rowwise().norm().array();

    MatrixXr XY = Xm * Ym.transpose();

    Eigen::Index n = X.rows();
    Eigen::Index m = Y.rows();
    ArrayXXr dist(n, m);

    Real eps = std::numeric_limits<Real>::epsilon();

    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < m; ++j) {
            Real denom = X_norm(i) * Y_norm(j);
            if (denom < eps) {
                dist(i, j) = 1.0;
            } else {
                dist(i, j) = 1.0 - XY(i, j) / denom;
            }
            // Clamp to [0, 2] to handle numerical issues
            dist(i, j) = std::max(Real(0.0), std::min(dist(i, j), Real(2.0)));
        }
    }

    return dist;
}

/// Build a k-nearest-neighbor graph from pairwise distances.
/// Returns a sparse matrix where entry (i, j) indicates j is a neighbor of i.
/// @param dist Pairwise distance matrix (n, m) — may be (n, n) for self-similarity
/// @param k Number of nearest neighbors per point
/// @param mode "connectivity" (binary), "distance" (raw distances)
/// @return Sparse matrix (n, m) in CSR format
inline SparseMatrixXr knn_graph(
    const ArrayXXr& dist,
    int k,
    const std::string& mode = "connectivity") {

    Eigen::Index n = dist.rows();
    Eigen::Index m = dist.cols();
    int actual_k = std::min(k, static_cast<int>(m));

    using Triplet = Eigen::Triplet<Real>;
    std::vector<Triplet> triplets;
    triplets.reserve(n * actual_k);

    // For each row, find k smallest distances
    std::vector<Eigen::Index> indices(m);

    for (Eigen::Index i = 0; i < n; ++i) {
        std::iota(indices.begin(), indices.end(), Eigen::Index(0));

        // Partial sort to find k smallest
        std::nth_element(indices.begin(), indices.begin() + actual_k,
                         indices.end(),
                         [&](Eigen::Index a, Eigen::Index b) {
                             return dist(i, a) < dist(i, b);
                         });

        for (int ki = 0; ki < actual_k; ++ki) {
            Eigen::Index j = indices[ki];
            if (mode == "connectivity") {
                triplets.emplace_back(i, j, 1.0);
            } else {
                triplets.emplace_back(i, j, dist(i, j));
            }
        }
    }

    SparseMatrixXr graph(n, m);
    graph.setFromTriplets(triplets.begin(), triplets.end());
    return graph;
}

/// Estimate affinity bandwidth using the "med_k_scalar" method.
/// Returns the median distance to the k-th nearest neighbor across all samples.
/// @param dist_graph Sparse distance graph (from knn_graph with mode="distance")
/// @param k Number of neighbors used for bandwidth estimation
/// @return Scalar bandwidth
inline Real affinity_bandwidth_med_k(const SparseMatrixXr& dist_graph, int k) {
    Eigen::Index n = dist_graph.rows();
    std::vector<Real> kth_dists;
    kth_dists.reserve(n);

    for (Eigen::Index i = 0; i < n; ++i) {
        // Collect all non-zero distances for row i
        std::vector<Real> row_dists;
        for (SparseMatrixXr::InnerIterator it(dist_graph, i); it; ++it) {
            if (it.value() > 0) {
                row_dists.push_back(it.value());
            }
        }

        if (row_dists.empty()) continue;

        std::sort(row_dists.begin(), row_dists.end());
        // k-th neighbor (0-indexed: k-1, but clamped)
        int idx = std::min(k - 1, static_cast<int>(row_dists.size()) - 1);
        kth_dists.push_back(row_dists[idx]);
    }

    if (kth_dists.empty()) {
        throw ParameterError("Cannot estimate bandwidth from an empty graph");
    }

    // Compute median
    size_t mid = kth_dists.size() / 2;
    std::nth_element(kth_dists.begin(), kth_dists.begin() + mid, kth_dists.end());
    if (kth_dists.size() % 2 == 0) {
        Real hi = kth_dists[mid];
        std::nth_element(kth_dists.begin(), kth_dists.begin() + mid - 1, kth_dists.end());
        return (kth_dists[mid - 1] + hi) / 2.0;
    }
    return kth_dists[mid];
}

} // namespace internal
} // namespace librosa
