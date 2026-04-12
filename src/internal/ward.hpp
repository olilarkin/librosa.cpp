#pragma once

#include <librosa/types.hpp>
#include <librosa/util/exceptions.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>

namespace librosa {
namespace internal {

/// Ward agglomerative clustering with 1D temporal connectivity.
/// Only adjacent clusters can merge. Returns segment boundaries.
/// @param data Feature matrix (n_samples, n_features) — rows are samples
/// @param k Number of clusters (segments) to produce
/// @return Sorted vector of left-boundary indices (always includes 0)
inline std::vector<Eigen::Index> ward_cluster_1d(
    const ArrayXXr& data,
    int k) {

    Eigen::Index n = data.rows();

    if (k <= 0) {
        throw ParameterError("k must be a positive integer");
    }

    if (k >= n) {
        // Each frame is its own segment
        std::vector<Eigen::Index> boundaries(n);
        std::iota(boundaries.begin(), boundaries.end(), 0);
        return boundaries;
    }

    if (k == 1) {
        return {0};
    }

    // Each cluster stores: start index, end index (exclusive), count, centroid sum
    struct Cluster {
        Eigen::Index start;
        Eigen::Index end;    // exclusive
        Eigen::Index count;
        ArrayXr sum;         // sum of all points in cluster (for centroid computation)
        int prev;            // index of previous active cluster (-1 if none)
        int next;            // index of next active cluster (-1 if none)
        bool active;
    };

    // Initialize: each sample is its own cluster
    std::vector<Cluster> clusters(n);
    Eigen::Index n_features = data.cols();

    for (Eigen::Index i = 0; i < n; ++i) {
        clusters[i].start = i;
        clusters[i].end = i + 1;
        clusters[i].count = 1;
        clusters[i].sum = data.row(i).transpose();
        clusters[i].prev = static_cast<int>(i) - 1;
        clusters[i].next = (i < n - 1) ? static_cast<int>(i) + 1 : -1;
        clusters[i].active = true;
    }

    // Compute Ward distance between two adjacent clusters
    auto ward_distance = [&](int a, int b) -> Real {
        Eigen::Index na = clusters[a].count;
        Eigen::Index nb = clusters[b].count;
        ArrayXr centroid_a = clusters[a].sum / static_cast<Real>(na);
        ArrayXr centroid_b = clusters[b].sum / static_cast<Real>(nb);
        Real diff_sq = (centroid_a - centroid_b).square().sum();
        return std::sqrt(static_cast<Real>(na * nb) / static_cast<Real>(na + nb)) *
               std::sqrt(diff_sq);
    };

    // Priority queue: (distance, left_idx, right_idx)
    // We use a version counter to invalidate stale entries
    struct MergeCandidate {
        Real distance;
        int left;
        int right;
        bool operator>(const MergeCandidate& other) const {
            return distance > other.distance;
        }
    };

    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>,
                        std::greater<MergeCandidate>> pq;

    // Initialize priority queue with all adjacent pairs
    for (Eigen::Index i = 0; i < n - 1; ++i) {
        Real d = ward_distance(static_cast<int>(i), static_cast<int>(i + 1));
        pq.push({d, static_cast<int>(i), static_cast<int>(i + 1)});
    }

    int n_clusters = static_cast<int>(n);
    int next_id = static_cast<int>(n);  // For new merged clusters

    while (n_clusters > k && !pq.empty()) {
        auto [dist, left, right] = pq.top();
        pq.pop();

        // Skip stale entries (clusters already merged)
        if (!clusters[left].active || !clusters[right].active) continue;
        // Verify they are still adjacent
        if (clusters[left].next != right) continue;

        // Merge right into left
        clusters[left].end = clusters[right].end;
        clusters[left].count += clusters[right].count;
        clusters[left].sum += clusters[right].sum;

        // Deactivate right
        clusters[right].active = false;

        // Update linked list
        int right_next = clusters[right].next;
        clusters[left].next = right_next;
        if (right_next >= 0) {
            clusters[right_next].prev = left;
        }

        n_clusters--;

        // Add new merge candidates with neighbors
        if (right_next >= 0 && clusters[right_next].active) {
            Real d = ward_distance(left, right_next);
            pq.push({d, left, right_next});
        }

        int left_prev = clusters[left].prev;
        if (left_prev >= 0 && clusters[left_prev].active) {
            Real d = ward_distance(left_prev, left);
            pq.push({d, left_prev, left});
        }
    }

    // Extract boundaries from active clusters
    std::vector<Eigen::Index> boundaries;
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(clusters.size()); ++i) {
        if (clusters[i].active) {
            boundaries.push_back(clusters[i].start);
        }
    }

    std::sort(boundaries.begin(), boundaries.end());
    return boundaries;
}

} // namespace internal
} // namespace librosa
