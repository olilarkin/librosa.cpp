#pragma once

#include "types.hpp"
#include <optional>
#include <string>
#include <vector>

namespace librosa {
namespace sequence {

// ============================================================================
// Viterbi Decoding
// ============================================================================

/// Viterbi decoding from observation likelihoods
/// @param prob Observation probabilities [shape: (n_states, n_steps)]
/// @param transition Transition matrix [shape: (n_states, n_states)]
/// @param p_init Initial state distribution (optional)
/// @param return_logp If true, return log-probability along with states
/// @return Most likely state sequence
std::vector<int> viterbi(
    const ArrayXXr& prob,
    const ArrayXXr& transition,
    std::optional<ArrayXr> p_init = std::nullopt);

/// Viterbi decoding with log-probability return
/// @param prob Observation probabilities [shape: (n_states, n_steps)]
/// @param transition Transition matrix [shape: (n_states, n_states)]
/// @param p_init Initial state distribution (optional)
/// @return Pair of (most likely state sequence, log-probability)
std::pair<std::vector<int>, Real> viterbi_with_logp(
    const ArrayXXr& prob,
    const ArrayXXr& transition,
    std::optional<ArrayXr> p_init = std::nullopt);

/// Viterbi decoding from discriminative state predictions
/// @param prob Conditional state probabilities [shape: (n_states, n_steps)]
/// @param transition Transition matrix [shape: (n_states, n_states)]
/// @param p_state Marginal state probability (optional)
/// @param p_init Initial state distribution (optional)
/// @return Most likely state sequence
std::vector<int> viterbi_discriminative(
    const ArrayXXr& prob,
    const ArrayXXr& transition,
    std::optional<ArrayXr> p_state = std::nullopt,
    std::optional<ArrayXr> p_init = std::nullopt);

// ============================================================================
// Transition Matrices
// ============================================================================

/// Construct a uniform transition matrix
/// @param n_states Number of states
/// @return Transition matrix where all entries are 1/n_states
ArrayXXr transition_uniform(int n_states);

/// Construct a self-loop transition matrix
/// @param n_states Number of states
/// @param prob Self-transition probability (scalar or per-state)
/// @return Transition matrix with self-loops
ArrayXXr transition_loop(int n_states, Real prob);
ArrayXXr transition_loop(int n_states, const ArrayXr& prob);

/// Construct a cyclic transition matrix
/// @param n_states Number of states
/// @param prob Self-transition probability (scalar or per-state)
/// @return Cyclic transition matrix
ArrayXXr transition_cycle(int n_states, Real prob);
ArrayXXr transition_cycle(int n_states, const ArrayXr& prob);

/// Construct a localized transition matrix
/// @param n_states Number of states
/// @param width Locality width
/// @param window Window type for locality shape
/// @param wrap If true, wrap around boundaries
/// @return Localized transition matrix
ArrayXXr transition_local(
    int n_states,
    int width,
    WindowType window = WindowType::Triangle,
    bool wrap = false);

/// Viterbi decoding for binary (multi-label) discriminative predictions
/// Transforms multi-label decoding into a collection of binary Viterbi problems.
/// @param prob State probabilities [shape: (n_states, n_steps)], values in [0, 1]
/// @param transition Transition matrix [shape: (2, 2)] or [shape: (n_states, 2, 2)]
///        For 2x2: same matrix applied to each state label
///        For 3D: per-state transition matrices (flattened as (n_states*2, 2))
/// @param p_state Marginal state probability (optional, default 0.5 each)
/// @param p_init Initial state probability (optional, default 0.5 each)
/// @return Binary state matrix [shape: (n_states, n_steps)]
ArrayXXr viterbi_binary(
    const ArrayXXr& prob,
    const ArrayXXr& transition,
    std::optional<ArrayXr> p_state = std::nullopt,
    std::optional<ArrayXr> p_init = std::nullopt);

/// Viterbi binary with log-probability return
/// @return Pair of (state matrix, log-probabilities per state)
std::pair<ArrayXXr, ArrayXr> viterbi_binary_with_logp(
    const ArrayXXr& prob,
    const ArrayXXr& transition,
    std::optional<ArrayXr> p_state = std::nullopt,
    std::optional<ArrayXr> p_init = std::nullopt);

// ============================================================================
// Recurrence Quantification Analysis
// ============================================================================

/// Recurrence quantification analysis
/// Computes alignment scores via dynamic programming on a similarity matrix.
/// Supports diagonal moves, knight moves, and gap penalties.
/// @param sim Similarity matrix [shape: (N, M)], non-negative
/// @param gap_onset Penalty for introducing a gap (default 1)
/// @param gap_extend Penalty for extending a gap (default 1)
/// @param knight_moves Allow knight moves in alignment (default true)
/// @param backtrack If true, return alignment path (default true)
/// @return Score matrix (and optionally the path)
ArrayXXr rqa(
    const ArrayXXr& sim,
    Real gap_onset = 1.0,
    Real gap_extend = 1.0,
    bool knight_moves = true);

/// RQA with backtracking
/// @return Pair of (score matrix, alignment path as vector of (row, col) pairs)
std::pair<ArrayXXr, std::vector<std::pair<int, int>>> rqa_backtrack(
    const ArrayXXr& sim,
    Real gap_onset = 1.0,
    Real gap_extend = 1.0,
    bool knight_moves = true);

// ============================================================================
// Dynamic Time Warping
// ============================================================================

/// Compute pairwise Euclidean distance between columns of X and Y
/// @param X First feature matrix [shape: (k, n)]
/// @param Y Second feature matrix [shape: (k, m)]
/// @return Distance matrix [shape: (n, m)]
ArrayXXr cdist_euclidean(const ArrayXXr& X, const ArrayXXr& Y);

/// Compute pairwise cosine distance between columns of X and Y
/// cosine_dist(i,j) = 1 - dot(X[:,i], Y[:,j]) / (||X[:,i]|| * ||Y[:,j]||)
/// @param X First feature matrix [shape: (k, n)]
/// @param Y Second feature matrix [shape: (k, m)]
/// @return Distance matrix [shape: (n, m)]
ArrayXXr cdist_cosine(const ArrayXXr& X, const ArrayXXr& Y);

/// Dynamic time warping
/// @param X First feature matrix [shape: (k, n)] or precomputed cost matrix C
/// @param Y Second feature matrix [shape: (k, m)] (optional if C provided)
/// @param subseq Enable subsequence DTW
/// @param backtrack Enable path backtracking
/// @return Accumulated cost matrix (and warping path if backtrack=true)
ArrayXXr dtw(
    const ArrayXXr& X,
    const ArrayXXr& Y,
    bool subseq = false,
    const std::string& metric = "euclidean");

/// Dynamic time warping with backtracking
/// @param X First feature matrix
/// @param Y Second feature matrix
/// @param subseq Enable subsequence DTW
/// @return Pair of (accumulated cost matrix, warping path)
std::pair<ArrayXXr, std::vector<std::pair<int, int>>> dtw_backtrack(
    const ArrayXXr& X,
    const ArrayXXr& Y,
    bool subseq = false,
    const std::string& metric = "euclidean");

/// DTW backtracking from a step matrix
/// @param steps Step matrix from DTW computation
/// @param subseq Enable subsequence DTW
/// @param start Starting column for backtracking (optional, for subseq)
/// @return Warping path as vector of (row, col) pairs
std::vector<std::pair<int, int>> dtw_backtracking(
    const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>& steps,
    bool subseq = false,
    std::optional<int> start = std::nullopt);

} // namespace sequence
} // namespace librosa
