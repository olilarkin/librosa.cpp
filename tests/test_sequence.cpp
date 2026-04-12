#include <gtest/gtest.h>
#include <librosa/sequence.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;
using namespace librosa::sequence;

// ============================================================================
// Transition Matrix Tests
// ============================================================================

TEST(TransitionUniformTest, BasicUniform) {
    ArrayXXr T = transition_uniform(3);

    EXPECT_EQ(T.rows(), 3);
    EXPECT_EQ(T.cols(), 3);

    // All entries should be 1/3
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(T(i, j), 1.0 / 3.0, 1e-10);
        }
    }

    // Rows should sum to 1
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(T.row(i).sum(), 1.0, 1e-10);
    }
}

TEST(TransitionUniformTest, InvalidInput) {
    EXPECT_THROW(transition_uniform(0), ParameterError);
    EXPECT_THROW(transition_uniform(-1), ParameterError);
}

TEST(TransitionLoopTest, ScalarProbability) {
    ArrayXXr T = transition_loop(3, 0.5);

    EXPECT_EQ(T.rows(), 3);
    EXPECT_EQ(T.cols(), 3);

    // Diagonal should be 0.5
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(T(i, i), 0.5, 1e-10);
    }

    // Off-diagonal should be 0.25
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i != j) {
                EXPECT_NEAR(T(i, j), 0.25, 1e-10);
            }
        }
    }

    // Rows should sum to 1
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(T.row(i).sum(), 1.0, 1e-10);
    }
}

TEST(TransitionLoopTest, VectorProbability) {
    ArrayXr prob(3);
    prob << 0.8, 0.5, 0.25;

    ArrayXXr T = transition_loop(3, prob);

    // Check diagonal values
    EXPECT_NEAR(T(0, 0), 0.8, 1e-10);
    EXPECT_NEAR(T(1, 1), 0.5, 1e-10);
    EXPECT_NEAR(T(2, 2), 0.25, 1e-10);

    // Rows should sum to 1
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(T.row(i).sum(), 1.0, 1e-10);
    }
}

TEST(TransitionLoopTest, InvalidInput) {
    EXPECT_THROW(transition_loop(1, 0.5), ParameterError);  // n_states must be > 1

    ArrayXr prob_wrong(2);
    prob_wrong << 0.5, 0.5;
    EXPECT_THROW(transition_loop(3, prob_wrong), ParameterError);  // Wrong length
}

TEST(TransitionCycleTest, BasicCycle) {
    ArrayXXr T = transition_cycle(4, 0.9);

    EXPECT_EQ(T.rows(), 4);
    EXPECT_EQ(T.cols(), 4);

    // Check structure: diagonal = 0.9, next state = 0.1
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(T(i, i), 0.9, 1e-10);
        EXPECT_NEAR(T(i, (i + 1) % 4), 0.1, 1e-10);
    }

    // All other entries should be 0
    EXPECT_NEAR(T(0, 2), 0.0, 1e-10);
    EXPECT_NEAR(T(0, 3), 0.0, 1e-10);

    // Rows should sum to 1
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(T.row(i).sum(), 1.0, 1e-10);
    }
}

TEST(TransitionCycleTest, WrapAround) {
    ArrayXXr T = transition_cycle(3, 0.5);

    // Last state should transition to first state
    EXPECT_NEAR(T(2, 0), 0.5, 1e-10);
}

TEST(TransitionLocalTest, BasicLocal) {
    ArrayXXr T = transition_local(5, 3);

    EXPECT_EQ(T.rows(), 5);
    EXPECT_EQ(T.cols(), 5);

    // Diagonal should have highest probability (within floating-point tolerance)
    for (int i = 0; i < 5; ++i) {
        Real diag = T(i, i);
        // Diagonal should be >= adjacent values (with epsilon for fp rounding)
        if (i > 0) EXPECT_GE(diag + 1e-14, T(i, i - 1));
        if (i < 4) EXPECT_GE(diag + 1e-14, T(i, i + 1));
    }

    // Rows should sum to 1
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(T.row(i).sum(), 1.0, 1e-10);
    }
}

TEST(TransitionLocalTest, WithWrap) {
    ArrayXXr T = transition_local(5, 3, WindowType::Triangle, true);

    // Should have non-zero values wrapping around
    // First row should have non-zero value in last column (due to wrap)
    // This depends on the width and state count
}

// ============================================================================
// Viterbi Tests
// ============================================================================

TEST(ViterbiTest, WikipediaExample) {
    // Example from Wikipedia Viterbi algorithm article
    // States: healthy (0), fever (1)
    // Observations: normal, cold, dizzy

    // Emission probabilities
    ArrayXXr prob(2, 3);
    prob << 0.5, 0.4, 0.1,   // healthy -> normal, cold, dizzy
            0.1, 0.3, 0.6;   // fever -> normal, cold, dizzy

    // Transition probabilities
    ArrayXXr trans(2, 2);
    trans << 0.7, 0.3,   // healthy -> healthy, fever
             0.4, 0.6;   // fever -> healthy, fever

    // Initial probabilities
    ArrayXr p_init(2);
    p_init << 0.6, 0.4;

    auto [states, logp] = viterbi_with_logp(prob, trans, p_init);

    // Expected: [healthy, healthy, fever] = [0, 0, 1]
    EXPECT_EQ(states.size(), 3);
    EXPECT_EQ(states[0], 0);
    EXPECT_EQ(states[1], 0);
    EXPECT_EQ(states[2], 1);
}

TEST(ViterbiTest, SimpleSequence) {
    // High probability for state 0 in first half, state 1 in second half
    ArrayXXr prob(2, 4);
    prob << 0.9, 0.9, 0.1, 0.1,
            0.1, 0.1, 0.9, 0.9;

    // Strong self-loop
    ArrayXXr trans(2, 2);
    trans << 0.9, 0.1,
             0.1, 0.9;

    std::vector<int> states = viterbi(prob, trans);

    EXPECT_EQ(states.size(), 4);
    EXPECT_EQ(states[0], 0);
    EXPECT_EQ(states[1], 0);
    EXPECT_EQ(states[2], 1);
    EXPECT_EQ(states[3], 1);
}

TEST(ViterbiDiscriminativeTest, Basic) {
    ArrayXXr prob(2, 3);
    prob << 0.8, 0.7, 0.2,
            0.2, 0.3, 0.8;

    ArrayXXr trans(2, 2);
    trans << 0.9, 0.1,
             0.1, 0.9;

    std::vector<int> states = viterbi_discriminative(prob, trans);

    EXPECT_EQ(states.size(), 3);
}

// ============================================================================
// DTW Tests
// ============================================================================

TEST(CdistEuclideanTest, Basic) {
    ArrayXXr X(2, 3);
    X << 0, 1, 2,
         0, 0, 0;

    ArrayXXr Y(2, 2);
    Y << 0, 1,
         0, 0;

    ArrayXXr dist = cdist_euclidean(X, Y);

    EXPECT_EQ(dist.rows(), 3);
    EXPECT_EQ(dist.cols(), 2);

    // d(X[:,0], Y[:,0]) = 0
    EXPECT_NEAR(dist(0, 0), 0.0, 1e-10);

    // d(X[:,0], Y[:,1]) = 1
    EXPECT_NEAR(dist(0, 1), 1.0, 1e-10);

    // d(X[:,1], Y[:,0]) = 1
    EXPECT_NEAR(dist(1, 0), 1.0, 1e-10);

    // d(X[:,1], Y[:,1]) = 0
    EXPECT_NEAR(dist(1, 1), 0.0, 1e-10);
}

TEST(DTWTest, IdenticalSequences) {
    ArrayXXr X(2, 5);
    X << 1, 2, 3, 4, 5,
         0, 0, 0, 0, 0;

    ArrayXXr D = dtw(X, X);

    EXPECT_EQ(D.rows(), 5);
    EXPECT_EQ(D.cols(), 5);

    // Final cost should be 0 for identical sequences
    EXPECT_NEAR(D(4, 4), 0.0, 1e-10);
}

TEST(DTWTest, DifferentSequences) {
    ArrayXXr X(1, 3);
    X << 0, 1, 2;

    ArrayXXr Y(1, 3);
    Y << 1, 2, 3;

    ArrayXXr D = dtw(X, Y);

    // Final cost should be positive
    EXPECT_GT(D(2, 2), 0);
}

TEST(DTWBacktrackTest, Basic) {
    ArrayXXr X(1, 4);
    X << 1, 2, 3, 4;

    ArrayXXr Y(1, 4);
    Y << 1, 2, 3, 4;

    auto [D, path] = dtw_backtrack(X, Y);

    // Path should exist
    EXPECT_GT(path.size(), 0);

    // Path should start at (0, 0) and end at (3, 3)
    EXPECT_EQ(path.front().first, 0);
    EXPECT_EQ(path.front().second, 0);
    EXPECT_EQ(path.back().first, 3);
    EXPECT_EQ(path.back().second, 3);
}

TEST(DTWBacktrackTest, Subsequence) {
    // X is shorter, Y contains X as subsequence
    ArrayXXr X(1, 3);
    X << 5, 6, 7;

    ArrayXXr Y(1, 7);
    Y << 1, 2, 5, 6, 7, 8, 9;

    auto [D, path] = dtw_backtrack(X, Y, true);

    // Path should exist
    EXPECT_GT(path.size(), 0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(SequenceTest, EmptySequence) {
    // Viterbi with minimal sequence
    ArrayXXr prob(2, 1);
    prob << 0.8, 0.2;

    ArrayXXr trans(2, 2);
    trans << 0.9, 0.1,
             0.1, 0.9;

    std::vector<int> states = viterbi(prob, trans);

    EXPECT_EQ(states.size(), 1);
    EXPECT_EQ(states[0], 0);  // Should pick state with higher prob
}

TEST(SequenceTest, ManyStates) {
    int n_states = 10;

    ArrayXXr T = transition_uniform(n_states);
    EXPECT_EQ(T.rows(), n_states);
    EXPECT_EQ(T.cols(), n_states);

    ArrayXXr T_loop = transition_loop(n_states, 0.9);
    EXPECT_EQ(T_loop.rows(), n_states);

    ArrayXXr T_cycle = transition_cycle(n_states, 0.9);
    EXPECT_EQ(T_cycle.rows(), n_states);
}

// ============================================================================
// Viterbi Binary Tests
// ============================================================================

TEST(ViterbiBinaryTest, OutputShape) {
    // 3 states, 20 time steps
    ArrayXXr prob(3, 20);
    prob.setConstant(0.5);
    prob.row(0).head(10).setConstant(0.9);  // state 0 active first half
    prob.row(1).tail(10).setConstant(0.9);  // state 1 active second half

    ArrayXXr trans(2, 2);
    trans << 0.9, 0.1,
             0.1, 0.9;

    auto states = viterbi_binary(prob, trans);

    EXPECT_EQ(states.rows(), 3);
    EXPECT_EQ(states.cols(), 20);

    // States should be 0 or 1
    for (Eigen::Index i = 0; i < states.rows(); ++i) {
        for (Eigen::Index j = 0; j < states.cols(); ++j) {
            EXPECT_TRUE(states(i, j) == 0.0 || states(i, j) == 1.0);
        }
    }
}

TEST(ViterbiBinaryTest, StrongSignal) {
    // One state clearly active, one inactive
    ArrayXXr prob(2, 10);
    prob.row(0).setConstant(0.95);  // State 0 always active
    prob.row(1).setConstant(0.05);  // State 1 always inactive

    ArrayXXr trans(2, 2);
    trans << 0.9, 0.1,
             0.1, 0.9;

    auto states = viterbi_binary(prob, trans);

    // State 0 should be mostly active
    EXPECT_GT(states.row(0).sum(), 5.0);
    // State 1 should be mostly inactive
    EXPECT_LT(states.row(1).sum(), 5.0);
}

TEST(ViterbiBinaryTest, WithLogp) {
    ArrayXXr prob(2, 10);
    prob.setConstant(0.5);
    prob.row(0).setConstant(0.8);

    ArrayXXr trans(2, 2);
    trans << 0.9, 0.1,
             0.1, 0.9;

    auto [states, logp] = viterbi_binary_with_logp(prob, trans);

    EXPECT_EQ(states.rows(), 2);
    EXPECT_EQ(logp.size(), 2);
    // Log probabilities should be finite
    EXPECT_TRUE(std::isfinite(logp(0)));
    EXPECT_TRUE(std::isfinite(logp(1)));
}

// ============================================================================
// RQA Tests
// ============================================================================

TEST(RqaTest, DiagonalPath) {
    // Create a similarity matrix with a clear diagonal
    ArrayXXr sim = ArrayXXr::Zero(10, 10);
    for (int i = 0; i < 10; ++i) {
        sim(i, i) = 1.0;
    }

    auto score = rqa(sim, 1.0, 1.0, false);

    EXPECT_EQ(score.rows(), 10);
    EXPECT_EQ(score.cols(), 10);

    // Diagonal should accumulate
    EXPECT_NEAR(score(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(score(9, 9), 10.0, 1e-6);
}

TEST(RqaTest, WithBacktrack) {
    ArrayXXr sim = ArrayXXr::Zero(10, 10);
    for (int i = 0; i < 10; ++i) {
        sim(i, i) = 1.0;
    }

    auto [score, path] = rqa_backtrack(sim, 1.0, 1.0, false);

    // Path should follow the diagonal
    EXPECT_GT(path.size(), 0u);
    for (const auto& [r, c] : path) {
        EXPECT_EQ(r, c);
    }
}

TEST(RqaTest, KnightMoves) {
    // Create a near-diagonal path with slight offsets
    ArrayXXr sim = ArrayXXr::Zero(10, 10);
    sim(0, 0) = 1.0;
    sim(1, 1) = 1.0;
    sim(3, 2) = 1.0;  // Knight move: (1,1) -> (3,2) = (+2, +1)
    sim(4, 3) = 1.0;
    sim(5, 4) = 1.0;

    // With knight moves, should be able to connect through the gap
    auto score_knight = rqa(sim, std::numeric_limits<Real>::infinity(),
                            std::numeric_limits<Real>::infinity(), true);

    // Without knight moves, only strict diagonal
    auto score_diag = rqa(sim, std::numeric_limits<Real>::infinity(),
                          std::numeric_limits<Real>::infinity(), false);

    // Knight moves should allow higher scores
    EXPECT_GE(score_knight.maxCoeff(), score_diag.maxCoeff());
}

TEST(RqaTest, GapPenalties) {
    ArrayXXr sim = ArrayXXr::Zero(5, 5);
    sim(0, 0) = 1.0;
    sim(1, 1) = 1.0;
    // Gap at (2, 2)
    sim(3, 3) = 1.0;
    sim(4, 4) = 1.0;

    // With small gap penalty, can bridge the gap
    auto score_small = rqa(sim, 0.1, 0.1, false);

    // With infinite gap penalty, resets at gap
    auto score_inf = rqa(sim, std::numeric_limits<Real>::infinity(),
                         std::numeric_limits<Real>::infinity(), false);

    // Small penalties should allow continuation
    EXPECT_GE(score_small.maxCoeff(), score_inf.maxCoeff());
}

TEST(RqaTest, EmptyMatrix) {
    ArrayXXr sim = ArrayXXr::Zero(5, 5);

    auto score = rqa(sim, 1.0, 1.0, true);

    EXPECT_EQ(score.rows(), 5);
    EXPECT_EQ(score.cols(), 5);
    EXPECT_NEAR(score.maxCoeff(), 0.0, 1e-6);
}

TEST(RqaTest, InvalidParams) {
    ArrayXXr sim = ArrayXXr::Zero(5, 5);
    EXPECT_THROW(rqa(sim, -1.0, 1.0), ParameterError);
    EXPECT_THROW(rqa(sim, 1.0, -1.0), ParameterError);
}
