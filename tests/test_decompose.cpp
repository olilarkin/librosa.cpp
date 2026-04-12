#include <gtest/gtest.h>
#include <librosa/decompose.hpp>
#include <librosa/segment.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;
using namespace librosa::decompose;

// ============================================================================
// Median Filter Tests
// ============================================================================

TEST(MedianFilter2DTest, BasicMedian3x3) {
    ArrayXXr S(5, 5);
    S << 1, 2, 3, 4, 5,
         2, 3, 4, 5, 6,
         3, 4, 5, 6, 7,
         4, 5, 6, 7, 8,
         5, 6, 7, 8, 9;

    ArrayXXr result = median_filter_2d(S, {3, 3}, "reflect");

    EXPECT_EQ(result.rows(), 5);
    EXPECT_EQ(result.cols(), 5);

    // Center element should remain 5 (median of 3x3 around it)
    EXPECT_NEAR(result(2, 2), 5.0, 1e-10);
}

TEST(MedianFilter2DTest, HorizontalFilter) {
    // Test horizontal filter (1 x width)
    ArrayXXr S(3, 7);
    S << 1, 2, 100, 4, 5, 6, 7,
         1, 2, 3, 100, 5, 6, 7,
         1, 2, 3, 4, 100, 6, 7;

    ArrayXXr result = median_filter_2d(S, {1, 3}, "reflect");

    // The spike (100) should be filtered out
    // Middle element of row 0: median of [2, 100, 4] = 4
    EXPECT_NEAR(result(0, 2), 4.0, 1e-10);
}

TEST(MedianFilter2DTest, VerticalFilter) {
    // Test vertical filter (height x 1)
    ArrayXXr S(7, 3);
    S << 1, 1, 1,
         2, 2, 2,
         100, 3, 3,
         4, 100, 4,
         5, 5, 100,
         6, 6, 6,
         7, 7, 7;

    ArrayXXr result = median_filter_2d(S, {3, 1}, "reflect");

    // Spikes should be filtered
    EXPECT_NEAR(result(2, 0), 4.0, 1e-10);  // median of [2, 100, 4] = 4
}

TEST(MedianFilter2DTest, ConstantMode) {
    ArrayXXr S(3, 3);
    S << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    ArrayXXr result = median_filter_2d(S, {3, 3}, "constant");

    EXPECT_EQ(result.rows(), 3);
    EXPECT_EQ(result.cols(), 3);
}

TEST(MedianFilter2DTest, EdgeMode) {
    ArrayXXr S(3, 3);
    S << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    ArrayXXr result = median_filter_2d(S, {3, 3}, "edge");

    EXPECT_EQ(result.rows(), 3);
    EXPECT_EQ(result.cols(), 3);
}

// ============================================================================
// HPSS Tests
// ============================================================================

TEST(HPSSTest, BasicSeparation) {
    // Create a simple spectrogram
    ArrayXXr S(10, 20);
    S.setRandom();
    S = S.abs();  // Make positive

    auto [H, P] = hpss(S, 5);

    EXPECT_EQ(H.rows(), S.rows());
    EXPECT_EQ(H.cols(), S.cols());
    EXPECT_EQ(P.rows(), S.rows());
    EXPECT_EQ(P.cols(), S.cols());

    // All values should be non-negative
    EXPECT_GE(H.minCoeff(), 0.0);
    EXPECT_GE(P.minCoeff(), 0.0);

    // H + P should roughly approximate S (for margin=1)
    ArrayXXr sum = H + P;
    Real max_diff = (sum - S).abs().maxCoeff();
    EXPECT_LT(max_diff, 1e-10);
}

TEST(HPSSTest, MaskOutput) {
    ArrayXXr S(10, 20);
    S.setRandom();
    S = S.abs();

    auto [mask_H, mask_P] = hpss(S, 5, 2.0, true);

    // Masks should be between 0 and 1
    EXPECT_GE(mask_H.minCoeff(), 0.0);
    EXPECT_LE(mask_H.maxCoeff(), 1.0);
    EXPECT_GE(mask_P.minCoeff(), 0.0);
    EXPECT_LE(mask_P.maxCoeff(), 1.0);

    // Masks should sum to 1 (for margin=1)
    ArrayXXr sum = mask_H + mask_P;
    for (Eigen::Index i = 0; i < sum.rows(); ++i) {
        for (Eigen::Index j = 0; j < sum.cols(); ++j) {
            EXPECT_NEAR(sum(i, j), 1.0, 1e-10);
        }
    }
}

TEST(HPSSTest, SeparateKernelSizes) {
    ArrayXXr S(10, 20);
    S.setRandom();
    S = S.abs();

    auto [H, P] = hpss(S, {7, 11});

    EXPECT_EQ(H.rows(), S.rows());
    EXPECT_EQ(H.cols(), S.cols());
}

TEST(HPSSTest, SeparateMargins) {
    ArrayXXr S(10, 20);
    S.setRandom();
    S = S.abs();

    auto [H, P] = hpss(S, 5, 2.0, false, {2.0, 3.0});

    EXPECT_EQ(H.rows(), S.rows());
    EXPECT_EQ(H.cols(), S.cols());

    // With margins > 1, H + P may not equal S (residual exists)
    ArrayXXr sum = H + P;
    // The sum should be <= S (with some tolerance for numerical errors)
    Real diff = (S - sum).minCoeff();
    EXPECT_GE(diff, -1e-10);
}

TEST(HPSSTest, HarmonicContent) {
    // Create spectrogram with horizontal stripes (harmonic content)
    ArrayXXr S(20, 50);
    S.setZero();

    // Add horizontal lines (constant across time)
    for (int f : {5, 10, 15}) {
        S.row(f).setConstant(1.0);
    }

    // Add some noise
    ArrayXXr noise = ArrayXXr::Random(20, 50) * 0.1;
    S = (S + noise.abs()).max(0.0);

    auto [H, P] = hpss(S, 11);

    // Harmonic component should preserve horizontal structure
    // Check that the harmonic component has higher energy at the harmonic rows
    Real harm_energy_at_5 = H.row(5).sum();
    Real harm_energy_at_3 = H.row(3).sum();  // Non-harmonic row

    // Harmonic rows should have more energy in H
    EXPECT_GT(harm_energy_at_5, harm_energy_at_3);
}

TEST(HPSSTest, PercussiveContent) {
    // Create spectrogram with vertical stripes (percussive content)
    ArrayXXr S(20, 50);
    S.setZero();

    // Add vertical lines (transient attacks)
    for (int t : {10, 25, 40}) {
        S.col(t).setConstant(1.0);
    }

    // Add some noise
    ArrayXXr noise = ArrayXXr::Random(20, 50) * 0.1;
    S = (S + noise.abs()).max(0.0);

    auto [H, P] = hpss(S, 11);

    // Percussive component should preserve vertical structure
    Real perc_energy_at_10 = P.col(10).sum();
    Real perc_energy_at_5 = P.col(5).sum();  // Non-percussive column

    // Percussive columns should have more energy in P
    EXPECT_GT(perc_energy_at_10, perc_energy_at_5);
}

TEST(HPSSTest, InvalidMargin) {
    ArrayXXr S(10, 20);
    S.setRandom();
    S = S.abs();

    // Margin < 1 should throw
    EXPECT_THROW(hpss(S, 5, 2.0, false, 0.5), ParameterError);
}

// ============================================================================
// Complex HPSS Tests
// ============================================================================

TEST(HPSSComplexTest, BasicSeparation) {
    // Create complex spectrogram
    ArrayXXc S(10, 20);
    ArrayXXr mag = ArrayXXr::Random(10, 20).abs() + 0.1;
    ArrayXXr phase = ArrayXXr::Random(10, 20) * M_PI;

    for (Eigen::Index i = 0; i < 10; ++i) {
        for (Eigen::Index j = 0; j < 20; ++j) {
            S(i, j) = std::polar(mag(i, j), phase(i, j));
        }
    }

    auto [H, P] = hpss_complex(S);

    EXPECT_EQ(H.rows(), S.rows());
    EXPECT_EQ(H.cols(), S.cols());
    EXPECT_EQ(P.rows(), S.rows());
    EXPECT_EQ(P.cols(), S.cols());
}

TEST(HPSSComplexTest, PreservesPhase) {
    // Create complex spectrogram with known phase
    ArrayXXc S(5, 10);
    for (Eigen::Index i = 0; i < 5; ++i) {
        for (Eigen::Index j = 0; j < 10; ++j) {
            Real mag = static_cast<Real>(i + j + 1);
            Real ph = static_cast<Real>(i * j) * 0.1;
            S(i, j) = std::polar(mag, ph);
        }
    }

    auto [H, P] = hpss_complex(S, 3, 2.0, false, 1.0);

    // Phase of H and P should match phase of S (where magnitude is non-zero)
    for (Eigen::Index i = 0; i < 5; ++i) {
        for (Eigen::Index j = 0; j < 10; ++j) {
            if (std::abs(H(i, j)) > 1e-10) {
                Real orig_phase = std::arg(S(i, j));
                Real harm_phase = std::arg(H(i, j));
                // Phases should be close (modulo 2*pi wrapping)
                Real phase_diff = std::abs(orig_phase - harm_phase);
                if (phase_diff > M_PI) phase_diff = 2 * M_PI - phase_diff;
                EXPECT_LT(phase_diff, 1e-6);
            }
        }
    }
}

TEST(HPSSComplexTest, MaskOutput) {
    ArrayXXc S(10, 20);
    ArrayXXr mag = ArrayXXr::Random(10, 20).abs() + 0.1;

    for (Eigen::Index i = 0; i < 10; ++i) {
        for (Eigen::Index j = 0; j < 20; ++j) {
            S(i, j) = Complex(mag(i, j), 0);
        }
    }

    auto [mask_H, mask_P] = hpss_complex(S, 5, 2.0, true);

    // For real input, masks should be real too
    for (Eigen::Index i = 0; i < 10; ++i) {
        for (Eigen::Index j = 0; j < 20; ++j) {
            EXPECT_NEAR(mask_H(i, j).imag(), 0.0, 1e-10);
            EXPECT_NEAR(mask_P(i, j).imag(), 0.0, 1e-10);
        }
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(HPSSTest, SmallSpectrogram) {
    ArrayXXr S(3, 3);
    S << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    auto [H, P] = hpss(S, 3);

    EXPECT_EQ(H.rows(), 3);
    EXPECT_EQ(H.cols(), 3);
}

TEST(HPSSTest, LargeKernel) {
    ArrayXXr S(5, 5);
    S.setRandom();
    S = S.abs();

    // Kernel larger than input dimension
    auto [H, P] = hpss(S, 11);

    EXPECT_EQ(H.rows(), 5);
    EXPECT_EQ(H.cols(), 5);
}

TEST(HPSSTest, DifferentPowers) {
    ArrayXXr S(10, 20);
    S.setRandom();
    S = S.abs();

    // Test different power values
    auto [H1, P1] = hpss(S, 5, 1.0);  // Linear
    auto [H2, P2] = hpss(S, 5, 2.0);  // Quadratic (default)
    auto [H4, P4] = hpss(S, 5, 4.0);  // Higher power

    // Higher power should give more binary-like masks
    // (but we're not returning masks here, so just check it runs)
    EXPECT_EQ(H1.rows(), S.rows());
    EXPECT_EQ(H2.rows(), S.rows());
    EXPECT_EQ(H4.rows(), S.rows());
}

// ============================================================================
// NMF Decomposition Tests
// ============================================================================

TEST(DecomposeNMFTest, BasicShape) {
    // Create a non-negative spectrogram
    ArrayXXr S = ArrayXXr::Random(20, 30).abs() + 0.01;

    auto [components, activations] = decompose_nmf(S, 5);

    EXPECT_EQ(components.rows(), 20);
    EXPECT_EQ(components.cols(), 5);
    EXPECT_EQ(activations.rows(), 5);
    EXPECT_EQ(activations.cols(), 30);
}

TEST(DecomposeNMFTest, NonNegative) {
    ArrayXXr S = ArrayXXr::Random(15, 25).abs() + 0.01;

    auto [components, activations] = decompose_nmf(S, 4);

    // Both factors should be non-negative
    EXPECT_GE(components.minCoeff(), 0.0);
    EXPECT_GE(activations.minCoeff(), 0.0);
}

TEST(DecomposeNMFTest, ReasonableReconstruction) {
    // Create a low-rank matrix: S = W * H
    ArrayXXr W_true = ArrayXXr::Random(20, 3).abs() + 0.1;
    ArrayXXr H_true = ArrayXXr::Random(3, 30).abs() + 0.1;
    ArrayXXr S = (W_true.matrix() * H_true.matrix()).array();

    auto [components, activations] = decompose_nmf(S, 3, false, 500, 1e-6);

    // Reconstruction should be close to original
    ArrayXXr S_hat = (components.matrix() * activations.matrix()).array();
    Real rel_error = (S - S_hat).matrix().norm() / S.matrix().norm();

    // Should achieve < 10% relative error on a rank-3 matrix with 3 components
    EXPECT_LT(rel_error, 0.1);
}

TEST(DecomposeNMFTest, DefaultComponents) {
    ArrayXXr S = ArrayXXr::Random(10, 20).abs() + 0.01;

    auto [components, activations] = decompose_nmf(S);

    // Default n_components = n_features
    EXPECT_EQ(components.cols(), 10);
}

TEST(DecomposeNMFTest, Sort) {
    ArrayXXr S = ArrayXXr::Random(20, 30).abs() + 0.01;

    auto [components, activations] = decompose_nmf(S, 5, true);

    EXPECT_EQ(components.rows(), 20);
    EXPECT_EQ(components.cols(), 5);
}

// ============================================================================
// Nearest-Neighbor Filter Tests
// ============================================================================

TEST(NNFilterTest, MeanFilter) {
    // Create feature matrix: 3 features, 5 time steps
    ArrayXXr S(3, 5);
    S << 1, 2, 3, 4, 5,
         1, 2, 3, 4, 5,
         1, 2, 3, 4, 5;

    // Create a recurrence matrix that connects each to its neighbor
    using Triplet = Eigen::Triplet<Real>;
    std::vector<Triplet> triplets;
    for (int i = 0; i < 5; ++i) {
        if (i > 0) triplets.emplace_back(i, i - 1, 1.0);
        if (i < 4) triplets.emplace_back(i, i + 1, 1.0);
    }
    SparseMatrixXr rec(5, 5);
    rec.setFromTriplets(triplets.begin(), triplets.end());

    ArrayXXr filtered = nn_filter(S, rec, AggregateFunc::Mean);

    EXPECT_EQ(filtered.rows(), 3);
    EXPECT_EQ(filtered.cols(), 5);

    // Interior points: mean of two neighbors
    // For column 2: neighbors are columns 1 and 3
    // Mean of (2,4) = 3, which equals original. But for column 1: neighbors are 0 and 2, mean = 2
    EXPECT_NEAR(filtered(0, 2), 3.0, 1e-10);
}

TEST(NNFilterTest, MedianFilter) {
    ArrayXXr S(1, 5);
    S << 1, 100, 2, 3, 4;

    // Connect each point to its ±1 neighbors
    using Triplet = Eigen::Triplet<Real>;
    std::vector<Triplet> triplets;
    for (int i = 0; i < 5; ++i) {
        if (i > 0) triplets.emplace_back(i, i - 1, 1.0);
        if (i < 4) triplets.emplace_back(i, i + 1, 1.0);
    }
    SparseMatrixXr rec(5, 5);
    rec.setFromTriplets(triplets.begin(), triplets.end());

    ArrayXXr filtered = nn_filter(S, rec, AggregateFunc::Median);

    EXPECT_EQ(filtered.rows(), 1);
    EXPECT_EQ(filtered.cols(), 5);

    // Column 1 (value=100): neighbors are 0 (value=1) and 2 (value=2)
    // Median of {1, 2} = 1.5
    EXPECT_NEAR(filtered(0, 1), 1.5, 1e-10);
}

TEST(NNFilterTest, EmptyNeighbors) {
    ArrayXXr S(2, 3);
    S << 1, 2, 3,
         4, 5, 6;

    // Empty recurrence matrix
    SparseMatrixXr rec(3, 3);

    ArrayXXr filtered = nn_filter(S, rec, AggregateFunc::Mean);

    // With no neighbors, output should equal input
    for (Eigen::Index i = 0; i < S.rows(); ++i) {
        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            EXPECT_NEAR(filtered(i, j), S(i, j), 1e-10);
        }
    }
}

TEST(NNFilterTest, InvalidShape) {
    ArrayXXr S(2, 5);
    S.setRandom();

    SparseMatrixXr rec(3, 3);  // Wrong size

    EXPECT_THROW(nn_filter(S, rec), ParameterError);
}
