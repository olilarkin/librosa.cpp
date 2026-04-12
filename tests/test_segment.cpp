#include <gtest/gtest.h>
#include <librosa/segment.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>
#include <numeric>

using namespace librosa;
using namespace librosa::segment;

// ============================================================================
// Recurrence to Lag and back
// ============================================================================

TEST(RecurrenceToLagTest, RoundTrip) {
    // Create a simple recurrence matrix with diagonal structure
    int n = 10;
    ArrayXXr rec = ArrayXXr::Zero(n, n);
    // Main diagonal
    for (int i = 0; i < n; ++i) {
        rec(i, i) = 1.0;
    }
    // Off-diagonal (lag=3)
    for (int i = 3; i < n; ++i) {
        rec(i, i - 3) = 1.0;
        rec(i - 3, i) = 1.0;
    }

    // Convert to lag and back
    ArrayXXr lag = recurrence_to_lag(rec, true);
    ArrayXXr rec2 = lag_to_recurrence(lag);

    // Should recover the original
    EXPECT_EQ(rec2.rows(), n);
    EXPECT_EQ(rec2.cols(), n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(rec2(i, j), rec(i, j), 1e-10);
        }
    }
}

TEST(RecurrenceToLagTest, NoPad) {
    int n = 8;
    ArrayXXr rec = MatrixXr::Identity(n, n).array();

    ArrayXXr lag = recurrence_to_lag(rec, false);

    // Without padding, lag should be square (n x n)
    EXPECT_EQ(lag.rows(), n);
    EXPECT_EQ(lag.cols(), n);

    // Main diagonal in recurrence -> row 0 in lag (all ones)
    for (int j = 0; j < n; ++j) {
        EXPECT_NEAR(lag(0, j), 1.0, 1e-10);
    }
}

TEST(RecurrenceToLagTest, PadShape) {
    int n = 8;
    ArrayXXr rec = MatrixXr::Identity(n, n).array();

    ArrayXXr lag = recurrence_to_lag(rec, true);

    // With padding, lag should have 2n rows (or cols, depending on axis)
    EXPECT_EQ(lag.rows(), 2 * n);
    EXPECT_EQ(lag.cols(), n);
}

TEST(RecurrenceToLagTest, NonSquareThrows) {
    ArrayXXr rec(3, 5);
    rec.setZero();
    EXPECT_THROW(recurrence_to_lag(rec), ParameterError);
}

// ============================================================================
// Cross-similarity
// ============================================================================

TEST(CrossSimilarityTest, BasicConnectivity) {
    // Create simple feature matrices with known structure
    // data: 3 features, 5 samples
    ArrayXXr data(3, 5);
    data << 1, 2, 3, 4, 5,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0;

    ArrayXXr data_ref(3, 5);
    data_ref << 1, 2, 3, 4, 5,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0;

    ArrayXXr xsim = cross_similarity(data, data_ref, 2, "euclidean", "connectivity");

    // Shape should be (n_ref=5, n=5)
    EXPECT_EQ(xsim.rows(), 5);
    EXPECT_EQ(xsim.cols(), 5);

    // Values should be 0 or 1
    for (Eigen::Index i = 0; i < xsim.rows(); ++i) {
        for (Eigen::Index j = 0; j < xsim.cols(); ++j) {
            EXPECT_TRUE(xsim(i, j) == 0.0 || xsim(i, j) == 1.0);
        }
    }
}

TEST(CrossSimilarityTest, Distance) {
    ArrayXXr data(2, 4);
    data << 0, 1, 2, 3,
            0, 0, 0, 0;

    ArrayXXr data_ref(2, 4);
    data_ref << 0, 1, 2, 3,
                0, 0, 0, 0;

    ArrayXXr xsim = cross_similarity(data, data_ref, 2, "euclidean", "distance");

    EXPECT_EQ(xsim.rows(), 4);
    EXPECT_EQ(xsim.cols(), 4);

    // Non-zero entries should contain actual distances
    // Self-similarity: closest neighbor to sample 0 should be sample 1 (dist=1)
}

TEST(CrossSimilarityTest, Affinity) {
    ArrayXXr data(2, 6);
    data << 0, 1, 2, 10, 11, 12,
            0, 0, 0, 0,  0,  0;

    ArrayXXr data_ref = data;

    ArrayXXr xsim = cross_similarity(data, data_ref, 2, "euclidean", "affinity");

    EXPECT_EQ(xsim.rows(), 6);
    EXPECT_EQ(xsim.cols(), 6);

    // Affinity values should be in [0, 1]
    EXPECT_GE(xsim.minCoeff(), 0.0);
    EXPECT_LE(xsim.maxCoeff(), 1.0 + 1e-10);
}

TEST(CrossSimilarityTest, CosineMetric) {
    ArrayXXr data(2, 3);
    data << 1, 0, 1,
            0, 1, 1;

    ArrayXXr data_ref = data;

    ArrayXXr xsim = cross_similarity(data, data_ref, 2, "cosine", "connectivity");

    EXPECT_EQ(xsim.rows(), 3);
    EXPECT_EQ(xsim.cols(), 3);
}

TEST(CrossSimilarityTest, InvalidMode) {
    ArrayXXr data(2, 3);
    data.setRandom();
    ArrayXXr data_ref = data;

    EXPECT_THROW(cross_similarity(data, data_ref, 2, "euclidean", "invalid"),
                 ParameterError);
}

// ============================================================================
// Recurrence Matrix
// ============================================================================

TEST(RecurrenceMatrixTest, BasicShape) {
    // Create features with repeating pattern
    int n = 20;
    ArrayXXr data(3, n);
    for (int i = 0; i < n; ++i) {
        data(0, i) = std::sin(2.0 * M_PI * i / 5.0);
        data(1, i) = std::cos(2.0 * M_PI * i / 5.0);
        data(2, i) = 0.0;
    }

    ArrayXXr rec = recurrence_matrix(data, 3, 1, "euclidean", false, "connectivity");

    EXPECT_EQ(rec.rows(), n);
    EXPECT_EQ(rec.cols(), n);

    // Should have recurrence at period=5
    // Values should be 0 or 1
    for (Eigen::Index i = 0; i < rec.rows(); ++i) {
        for (Eigen::Index j = 0; j < rec.cols(); ++j) {
            EXPECT_TRUE(rec(i, j) == 0.0 || rec(i, j) == 1.0);
        }
    }
}

TEST(RecurrenceMatrixTest, WidthExclusion) {
    int n = 15;
    ArrayXXr data = ArrayXXr::Random(2, n);

    int width = 3;
    ArrayXXr rec = recurrence_matrix(data, 3, width, "euclidean", false, "connectivity");

    // No connections within the exclusion zone
    for (int diag = -(width - 1); diag <= (width - 1); ++diag) {
        for (Eigen::Index i = 0; i < n; ++i) {
            Eigen::Index j = i + diag;
            if (j >= 0 && j < n) {
                EXPECT_EQ(rec(i, j), 0.0);
            }
        }
    }
}

TEST(RecurrenceMatrixTest, Symmetric) {
    int n = 15;
    ArrayXXr data = ArrayXXr::Random(3, n);

    ArrayXXr rec = recurrence_matrix(data, 4, 1, "euclidean", true, "connectivity");

    // Should be symmetric
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            EXPECT_NEAR(rec(i, j), rec(j, i), 1e-10);
        }
    }
}

TEST(RecurrenceMatrixTest, SelfLoop) {
    int n = 10;
    ArrayXXr data = ArrayXXr::Random(2, n);

    ArrayXXr rec = recurrence_matrix(data, 3, 1, "euclidean", false, "connectivity",
                                      0.0, true);

    // Diagonal should be populated
    for (Eigen::Index i = 0; i < n; ++i) {
        EXPECT_EQ(rec(i, i), 1.0);
    }
}

TEST(RecurrenceMatrixTest, NoSelfLoop) {
    int n = 10;
    ArrayXXr data = ArrayXXr::Random(2, n);

    ArrayXXr rec = recurrence_matrix(data, 3, 1, "euclidean", false, "connectivity",
                                      0.0, false);

    // Diagonal should be zero
    for (Eigen::Index i = 0; i < n; ++i) {
        EXPECT_EQ(rec(i, i), 0.0);
    }
}

// ============================================================================
// Path Enhancement
// ============================================================================

TEST(PathEnhanceTest, PreservesShape) {
    int n = 20;
    ArrayXXr R = ArrayXXr::Random(n, n).abs();

    ArrayXXr R_smooth = path_enhance(R, 7);

    EXPECT_EQ(R_smooth.rows(), n);
    EXPECT_EQ(R_smooth.cols(), n);
}

TEST(PathEnhanceTest, NonNegative) {
    int n = 20;
    ArrayXXr R = ArrayXXr::Random(n, n).abs();

    ArrayXXr R_smooth = path_enhance(R, 7, WindowType::Hann, 2.0, 0.0, 7, false, true);

    // With clip=true, output should be non-negative
    EXPECT_GE(R_smooth.minCoeff(), 0.0);
}

TEST(PathEnhanceTest, EnhancesDiagonal) {
    int n = 30;
    // Create a matrix with a clear diagonal
    ArrayXXr R = ArrayXXr::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        R(i, i) = 1.0;
    }
    // Add noise
    R += ArrayXXr::Random(n, n).abs() * 0.1;

    ArrayXXr R_smooth = path_enhance(R, 11, WindowType::Hann, 1.5, 0.0, 5, false, true);

    // Diagonal should still be relatively strong
    Real diag_mean = 0.0;
    Real off_diag_mean = 0.0;
    for (int i = 0; i < n; ++i) {
        diag_mean += R_smooth(i, i);
    }
    diag_mean /= n;

    int count = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                off_diag_mean += R_smooth(i, j);
                count++;
            }
        }
    }
    off_diag_mean /= count;

    EXPECT_GT(diag_mean, off_diag_mean);
}

TEST(PathEnhanceTest, InvalidRatios) {
    ArrayXXr R = ArrayXXr::Random(10, 10).abs();
    EXPECT_THROW(path_enhance(R, 5, WindowType::Hann, 1.0, 3.0), ParameterError);
}

// ============================================================================
// Agglomerative Clustering
// ============================================================================

TEST(AgglomerativeTest, ClearBoundary) {
    // Data with a clear jump: [1,1,1,5,5,5]
    ArrayXXr data(1, 6);
    data << 1, 1, 1, 5, 5, 5;

    auto bounds = agglomerative(data, 2);

    ASSERT_EQ(bounds.size(), 2u);
    EXPECT_EQ(bounds[0], 0);
    EXPECT_EQ(bounds[1], 3);
}

TEST(AgglomerativeTest, SingleCluster) {
    ArrayXXr data(1, 5);
    data << 1, 2, 3, 4, 5;

    auto bounds = agglomerative(data, 1);

    ASSERT_EQ(bounds.size(), 1u);
    EXPECT_EQ(bounds[0], 0);
}

TEST(AgglomerativeTest, AllClusters) {
    int n = 5;
    ArrayXXr data(1, n);
    data << 1, 2, 3, 4, 5;

    auto bounds = agglomerative(data, n);

    ASSERT_EQ(bounds.size(), static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(bounds[i], i);
    }
}

TEST(AgglomerativeTest, MultiFeature) {
    // 2D features with clear cluster structure
    ArrayXXr data(2, 8);
    data << 0, 0, 0, 0, 10, 10, 10, 10,
            0, 0, 0, 0, 10, 10, 10, 10;

    auto bounds = agglomerative(data, 2);

    ASSERT_EQ(bounds.size(), 2u);
    EXPECT_EQ(bounds[0], 0);
    EXPECT_EQ(bounds[1], 4);
}

// ============================================================================
// Subsegment
// ============================================================================

TEST(SubsegmentTest, BasicSubdivision) {
    // Simple feature matrix
    ArrayXXr data(1, 12);
    data << 1, 1, 1, 5, 5, 5, 2, 2, 2, 8, 8, 8;

    std::vector<Eigen::Index> frames = {0, 6};

    auto bounds = subsegment(data, frames, 2);

    // Should include 0 and some subdivisions
    EXPECT_GE(bounds.size(), 2u);
    EXPECT_EQ(bounds[0], 0);
}

TEST(SubsegmentTest, InvalidNSegments) {
    ArrayXXr data(1, 10);
    data.setRandom();
    std::vector<Eigen::Index> frames = {0, 5};

    EXPECT_THROW(subsegment(data, frames, 0), ParameterError);
}

// ============================================================================
// Sparse versions
// ============================================================================

TEST(CrossSimilaritySparseTest, BasicShape) {
    ArrayXXr data(2, 5);
    data.setRandom();
    ArrayXXr data_ref(2, 5);
    data_ref.setRandom();

    SparseMatrixXr xsim = cross_similarity_sparse(data, data_ref, 2);

    EXPECT_EQ(xsim.rows(), 5);
    EXPECT_EQ(xsim.cols(), 5);
}

TEST(RecurrenceMatrixSparseTest, BasicShape) {
    ArrayXXr data(2, 15);
    data.setRandom();

    SparseMatrixXr rec = recurrence_matrix_sparse(data, 3, 1);

    EXPECT_EQ(rec.rows(), 15);
    EXPECT_EQ(rec.cols(), 15);
}
