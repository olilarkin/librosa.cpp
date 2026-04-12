#include <librosa/sequence.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

namespace librosa {
namespace sequence {

// ============================================================================
// Viterbi Decoding
// ============================================================================

std::vector<int> viterbi(
    const ArrayXXr& prob,
    const ArrayXXr& transition,
    std::optional<ArrayXr> p_init) {

    auto [states, logp] = viterbi_with_logp(prob, transition, p_init);
    return states;
}

std::pair<std::vector<int>, Real> viterbi_with_logp(
    const ArrayXXr& prob,
    const ArrayXXr& transition,
    std::optional<ArrayXr> p_init) {

    Eigen::Index n_states = prob.rows();
    Eigen::Index n_steps = prob.cols();

    if (transition.rows() != n_states || transition.cols() != n_states) {
        throw ParameterError("Transition matrix must be square with dimension n_states");
    }

    // Initialize state distribution
    ArrayXr init;
    if (p_init.has_value()) {
        init = p_init.value();
    } else {
        init = ArrayXr::Constant(n_states, 1.0 / n_states);
    }

    // Work in log domain for numerical stability
    ArrayXXr log_prob = prob.max(util::tiny(Real(1.0))).log();
    ArrayXXr log_trans = transition.max(util::tiny(Real(1.0))).log();
    ArrayXr log_init = init.max(util::tiny(Real(1.0))).log();

    // Viterbi trellis and backpointers
    ArrayXXr trellis(n_states, n_steps);
    Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> backpointer(n_states, n_steps);

    // Initialize first column
    trellis.col(0) = log_init + log_prob.col(0);
    backpointer.col(0).setZero();

    // Forward pass
    for (Eigen::Index t = 1; t < n_steps; ++t) {
        for (Eigen::Index s = 0; s < n_states; ++s) {
            Real best_val = -std::numeric_limits<Real>::infinity();
            int best_prev = 0;

            for (Eigen::Index prev = 0; prev < n_states; ++prev) {
                Real val = trellis(prev, t - 1) + log_trans(prev, s);
                if (val > best_val) {
                    best_val = val;
                    best_prev = static_cast<int>(prev);
                }
            }

            trellis(s, t) = best_val + log_prob(s, t);
            backpointer(s, t) = best_prev;
        }
    }

    // Find best final state
    int best_state = 0;
    Real best_logp = trellis(0, n_steps - 1);
    for (Eigen::Index s = 1; s < n_states; ++s) {
        if (trellis(s, n_steps - 1) > best_logp) {
            best_logp = trellis(s, n_steps - 1);
            best_state = static_cast<int>(s);
        }
    }

    // Backtrack
    std::vector<int> states(n_steps);
    states[n_steps - 1] = best_state;
    for (Eigen::Index t = n_steps - 1; t > 0; --t) {
        states[t - 1] = backpointer(states[t], t);
    }

    return {states, best_logp};
}

std::vector<int> viterbi_discriminative(
    const ArrayXXr& prob,
    const ArrayXXr& transition,
    std::optional<ArrayXr> p_state,
    std::optional<ArrayXr> p_init) {

    Eigen::Index n_states = prob.rows();

    // Get marginal state probabilities
    ArrayXr p_s;
    if (p_state.has_value()) {
        p_s = p_state.value();
    } else {
        p_s = ArrayXr::Constant(n_states, 1.0 / n_states);
    }

    // Convert discriminative to generative probabilities
    // P(obs|state) ∝ P(state|obs) / P(state)
    ArrayXXr prob_obs(prob.rows(), prob.cols());
    for (Eigen::Index s = 0; s < n_states; ++s) {
        prob_obs.row(s) = prob.row(s) / (p_s(s) + util::tiny(p_s(s)));
    }

    return viterbi(prob_obs, transition, p_init);
}

// ============================================================================
// Transition Matrices
// ============================================================================

ArrayXXr transition_uniform(int n_states) {
    if (!util::is_positive_int(n_states)) {
        throw ParameterError("n_states must be a positive integer");
    }

    return ArrayXXr::Constant(n_states, n_states, 1.0 / n_states);
}

ArrayXXr transition_loop(int n_states, Real prob) {
    ArrayXr probs = ArrayXr::Constant(n_states, prob);
    return transition_loop(n_states, probs);
}

ArrayXXr transition_loop(int n_states, const ArrayXr& prob) {
    if (!util::is_positive_int(n_states) || n_states <= 1) {
        throw ParameterError("n_states must be a positive integer > 1");
    }

    if (prob.size() != n_states) {
        throw ParameterError("prob must have length n_states");
    }

    // Check probability bounds
    if ((prob < 0).any() || (prob > 1).any()) {
        throw ParameterError("prob values must be in [0, 1]");
    }

    ArrayXXr transition(n_states, n_states);

    for (int i = 0; i < n_states; ++i) {
        Real self_prob = prob(i);
        Real other_prob = (1.0 - self_prob) / (n_states - 1);

        for (int j = 0; j < n_states; ++j) {
            transition(i, j) = (i == j) ? self_prob : other_prob;
        }
    }

    return transition;
}

ArrayXXr transition_cycle(int n_states, Real prob) {
    ArrayXr probs = ArrayXr::Constant(n_states, prob);
    return transition_cycle(n_states, probs);
}

ArrayXXr transition_cycle(int n_states, const ArrayXr& prob) {
    if (!util::is_positive_int(n_states) || n_states <= 1) {
        throw ParameterError("n_states must be a positive integer > 1");
    }

    if (prob.size() != n_states) {
        throw ParameterError("prob must have length n_states");
    }

    if ((prob < 0).any() || (prob > 1).any()) {
        throw ParameterError("prob values must be in [0, 1]");
    }

    ArrayXXr transition = ArrayXXr::Zero(n_states, n_states);

    for (int i = 0; i < n_states; ++i) {
        transition(i, i) = prob(i);
        transition(i, (i + 1) % n_states) = 1.0 - prob(i);
    }

    return transition;
}

ArrayXXr transition_local(
    int n_states,
    int width,
    WindowType window,
    bool wrap) {

    if (!util::is_positive_int(n_states) || n_states <= 1) {
        throw ParameterError("n_states must be a positive integer > 1");
    }

    if (!util::is_positive_int(width)) {
        throw ParameterError("width must be a positive integer");
    }

    // Get the window
    ArrayXr win = get_window(window, width, true);

    // Normalize window
    win /= win.sum();

    ArrayXXr transition = ArrayXXr::Zero(n_states, n_states);

    int half_width = width / 2;

    for (int i = 0; i < n_states; ++i) {
        for (int k = 0; k < width; ++k) {
            int j = i - half_width + k;

            if (wrap) {
                j = ((j % n_states) + n_states) % n_states;
            }

            if (j >= 0 && j < n_states) {
                transition(i, j) = win(k);
            }
        }

        // Re-normalize row (in case some values were clipped)
        Real row_sum = transition.row(i).sum();
        if (row_sum > 0) {
            transition.row(i) /= row_sum;
        }
    }

    return transition;
}

// ============================================================================
// Viterbi Binary (Multi-label)
// ============================================================================

std::pair<ArrayXXr, ArrayXr> viterbi_binary_with_logp(
    const ArrayXXr& prob,
    const ArrayXXr& transition,
    std::optional<ArrayXr> p_state,
    std::optional<ArrayXr> p_init) {

    Eigen::Index n_states = prob.rows();
    Eigen::Index n_steps = prob.cols();

    // Validate transition shape: must be (2, 2) or (n_states * 2, 2)
    bool per_state_transition = false;
    if (transition.rows() == 2 && transition.cols() == 2) {
        per_state_transition = false;
    } else if (transition.rows() == n_states * 2 && transition.cols() == 2) {
        per_state_transition = true;
    } else {
        throw ParameterError("transition must be (2, 2) or (n_states*2, 2)");
    }

    // Set defaults
    ArrayXr ps(n_states);
    if (p_state.has_value()) {
        ps = p_state.value();
    } else {
        ps.setConstant(0.5);
    }

    ArrayXr pi(n_states);
    if (p_init.has_value()) {
        pi = p_init.value();
    } else {
        pi.setConstant(0.5);
    }

    ArrayXXr states(n_states, n_steps);
    ArrayXr logp(n_states);

    // Process each state independently as a binary Viterbi problem
    ArrayXXr prob_binary(2, n_steps);
    ArrayXr p_state_binary(2);
    ArrayXr p_init_binary(2);

    for (Eigen::Index s = 0; s < n_states; ++s) {
        prob_binary.row(0) = 1.0 - prob.row(s);
        prob_binary.row(1) = prob.row(s);

        p_state_binary(0) = 1.0 - ps(s);
        p_state_binary(1) = ps(s);

        p_init_binary(0) = 1.0 - pi(s);
        p_init_binary(1) = pi(s);

        // Get the transition for this state
        ArrayXXr trans(2, 2);
        if (per_state_transition) {
            trans = transition.block(s * 2, 0, 2, 2);
        } else {
            trans = transition;
        }

        auto result = viterbi_discriminative(prob_binary, trans,
                                              p_state_binary, p_init_binary);

        for (Eigen::Index t = 0; t < n_steps; ++t) {
            states(s, t) = static_cast<Real>(result[t]);
        }

        // Compute logp via a second pass with viterbi_with_logp on the generative form
        // Convert discriminative to generative
        ArrayXXr prob_gen(2, n_steps);
        for (int k = 0; k < 2; ++k) {
            prob_gen.row(k) = prob_binary.row(k) / (p_state_binary(k) + util::tiny(p_state_binary(k)));
        }
        auto [_, lp] = viterbi_with_logp(prob_gen, trans, p_init_binary);
        logp(s) = lp;
    }

    return {states, logp};
}

ArrayXXr viterbi_binary(
    const ArrayXXr& prob,
    const ArrayXXr& transition,
    std::optional<ArrayXr> p_state,
    std::optional<ArrayXr> p_init) {

    auto [states, logp] = viterbi_binary_with_logp(prob, transition, p_state, p_init);
    return states;
}

// ============================================================================
// Recurrence Quantification Analysis
// ============================================================================

namespace {

// RQA dynamic programming core
std::pair<ArrayXXr, Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
rqa_dp(const ArrayXXr& sim, Real gap_onset, Real gap_extend, bool knight) {
    Eigen::Index rows = sim.rows();
    Eigen::Index cols = sim.cols();

    ArrayXXr score = ArrayXXr::Zero(rows, cols);
    Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> backtrack =
        Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(rows, cols);

    // Backtracking rubric:
    //  0 => diagonal, 1 => knight up, 2 => knight left
    // -1 => reset (no inclusion), -2 => start of sequence (inclusion)

    int init_limit = knight ? 2 : 1;
    int limit = knight ? 3 : 1;

    // Initialize first row and column
    for (Eigen::Index i = 0; i < rows; ++i) {
        score(i, 0) = sim(i, 0);
        backtrack(i, 0) = sim(i, 0) > 0 ? -2 : -1;
    }
    for (Eigen::Index j = 0; j < cols; ++j) {
        score(0, j) = sim(0, j);
        backtrack(0, j) = sim(0, j) > 0 ? -2 : -1;
    }

    // 1-1 case: diagonal only
    if (rows > 1 && cols > 1) {
        if (sim(1, 1) > 0) {
            score(1, 1) = score(0, 0) + sim(1, 1);
            backtrack(1, 1) = 0;
        } else {
            bool link = sim(0, 0) > 0;
            Real val = score(0, 0) - (link ? gap_onset : gap_extend);
            score(1, 1) = std::max(0.0, val);
            backtrack(1, 1) = score(1, 1) > 0 ? 0 : -1;
        }
    }

    // Second row (i=1): diagonal + left-knight
    if (rows > 1) {
        for (Eigen::Index j = 2; j < cols; ++j) {
            Real sv[3] = {score(0, j-1), score(0, j-2), 0};
            bool tv[3] = {sim(0, j-1) > 0, sim(0, j-2) > 0, false};

            if (sim(1, j) > 0) {
                int best = 0;
                for (int k = 1; k < init_limit; ++k)
                    if (sv[k] > sv[best]) best = k;
                backtrack(1, j) = best;
                score(1, j) = sv[best] + sim(1, j);
            } else {
                Real vec[3];
                for (int k = 0; k < init_limit; ++k)
                    vec[k] = sv[k] - (tv[k] ? gap_onset : gap_extend);
                int best = 0;
                for (int k = 1; k < init_limit; ++k)
                    if (vec[k] > vec[best]) best = k;
                score(1, j) = std::max(0.0, vec[best]);
                backtrack(1, j) = score(1, j) > 0 ? best : -1;
            }
        }
    }

    // Second column (j=1): diagonal + up-knight
    if (cols > 1) {
        for (Eigen::Index i = 2; i < rows; ++i) {
            Real sv[3] = {score(i-1, 0), score(i-2, 0), 0};
            bool tv[3] = {sim(i-1, 0) > 0, sim(i-2, 0) > 0, false};

            if (sim(i, 1) > 0) {
                int best = 0;
                for (int k = 1; k < init_limit; ++k)
                    if (sv[k] > sv[best]) best = k;
                backtrack(i, 1) = best;
                score(i, 1) = sv[best] + sim(i, 1);
            } else {
                Real vec[3];
                for (int k = 0; k < init_limit; ++k)
                    vec[k] = sv[k] - (tv[k] ? gap_onset : gap_extend);
                int best = 0;
                for (int k = 1; k < init_limit; ++k)
                    if (vec[k] > vec[best]) best = k;
                score(i, 1) = std::max(0.0, vec[best]);
                backtrack(i, 1) = score(i, 1) > 0 ? best : -1;
            }
        }
    }

    // Fill rest of table
    for (Eigen::Index i = 2; i < rows; ++i) {
        for (Eigen::Index j = 2; j < cols; ++j) {
            Real sv[3] = {score(i-1, j-1), score(i-1, j-2), score(i-2, j-1)};
            bool tv[3] = {sim(i-1, j-1) > 0, sim(i-1, j-2) > 0, sim(i-2, j-1) > 0};

            if (sim(i, j) > 0) {
                int best = 0;
                for (int k = 1; k < limit; ++k)
                    if (sv[k] > sv[best]) best = k;
                backtrack(i, j) = best;
                score(i, j) = sv[best] + sim(i, j);
            } else {
                Real vec[3];
                for (int k = 0; k < limit; ++k)
                    vec[k] = sv[k] - (tv[k] ? gap_onset : gap_extend);
                int best = 0;
                for (int k = 1; k < limit; ++k)
                    if (vec[k] > vec[best]) best = k;
                score(i, j) = std::max(0.0, vec[best]);
                backtrack(i, j) = score(i, j) > 0 ? best : -1;
            }
        }
    }

    return {score, backtrack};
}

// RQA backtracking
std::vector<std::pair<int, int>> rqa_backtrack_path(
    const ArrayXXr& score,
    const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& pointers) {

    // Offsets: 0 => (-1,-1), 1 => (-1,-2), 2 => (-2,-1)
    const int offsets[3][2] = {{-1, -1}, {-1, -2}, {-2, -1}};

    // Find maximum score position
    Eigen::Index max_r = 0, max_c = 0;
    Real max_val = score(0, 0);
    for (Eigen::Index i = 0; i < score.rows(); ++i) {
        for (Eigen::Index j = 0; j < score.cols(); ++j) {
            if (score(i, j) > max_val) {
                max_val = score(i, j);
                max_r = i;
                max_c = j;
            }
        }
    }

    std::vector<std::pair<int, int>> path;
    int i = static_cast<int>(max_r);
    int j = static_cast<int>(max_c);

    while (true) {
        int bt = pointers(i, j);

        if (bt == -1) break;  // non-inclusive reset

        path.push_back({i, j});

        if (bt == -2) break;  // start of sequence

        i += offsets[bt][0];
        j += offsets[bt][1];
    }

    std::reverse(path.begin(), path.end());
    return path;
}

} // anonymous namespace

ArrayXXr rqa(
    const ArrayXXr& sim,
    Real gap_onset,
    Real gap_extend,
    bool knight_moves) {

    if (gap_onset < 0) throw ParameterError("gap_onset must be non-negative");
    if (gap_extend < 0) throw ParameterError("gap_extend must be non-negative");

    auto [score, pointers] = rqa_dp(sim, gap_onset, gap_extend, knight_moves);
    return score;
}

std::pair<ArrayXXr, std::vector<std::pair<int, int>>> rqa_backtrack(
    const ArrayXXr& sim,
    Real gap_onset,
    Real gap_extend,
    bool knight_moves) {

    if (gap_onset < 0) throw ParameterError("gap_onset must be non-negative");
    if (gap_extend < 0) throw ParameterError("gap_extend must be non-negative");

    auto [score, pointers] = rqa_dp(sim, gap_onset, gap_extend, knight_moves);
    auto path = rqa_backtrack_path(score, pointers);
    return {score, path};
}

// ============================================================================
// Dynamic Time Warping
// ============================================================================

ArrayXXr cdist_euclidean(const ArrayXXr& X, const ArrayXXr& Y) {
    Eigen::Index n = X.cols();
    Eigen::Index m = Y.cols();

    ArrayXXr dist(n, m);

    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < m; ++j) {
            dist(i, j) = std::sqrt((X.col(i) - Y.col(j)).square().sum());
        }
    }

    return dist;
}

ArrayXXr cdist_cosine(const ArrayXXr& X, const ArrayXXr& Y) {
    // X: (k, n), Y: (k, m) — columns are data points
    Eigen::Index n = X.cols();
    Eigen::Index m = Y.cols();

    // Compute column norms
    ArrayXr X_norms(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        X_norms(i) = std::sqrt(X.col(i).square().sum());
    }
    ArrayXr Y_norms(m);
    for (Eigen::Index j = 0; j < m; ++j) {
        Y_norms(j) = std::sqrt(Y.col(j).square().sum());
    }

    // Dot products: X^T @ Y, shape (n, m)
    MatrixXr XtY = X.matrix().transpose() * Y.matrix();

    ArrayXXr dist(n, m);
    Real eps = std::numeric_limits<Real>::epsilon();

    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < m; ++j) {
            Real denom = X_norms(i) * Y_norms(j);
            if (denom < eps) {
                dist(i, j) = 1.0;
            } else {
                dist(i, j) = 1.0 - XtY(i, j) / denom;
            }
            dist(i, j) = std::max(Real(0.0), std::min(dist(i, j), Real(2.0)));
        }
    }

    return dist;
}

namespace {

ArrayXXr cdist_for_metric(const ArrayXXr& X, const ArrayXXr& Y, const std::string& metric) {
    if (metric == "euclidean") {
        return cdist_euclidean(X, Y);
    }
    if (metric == "cosine") {
        return cdist_cosine(X, Y);
    }
    throw ParameterError("Unsupported DTW metric: " + metric);
}

} // namespace

ArrayXXr dtw(
    const ArrayXXr& X,
    const ArrayXXr& Y,
    bool subseq,
    const std::string& metric) {

    // Compute cost matrix
    ArrayXXr C = cdist_for_metric(X, Y, metric);

    Eigen::Index n = C.rows();
    Eigen::Index m = C.cols();

    // Initialize accumulated cost matrix
    ArrayXXr D = ArrayXXr::Constant(n + 1, m + 1, std::numeric_limits<Real>::infinity());

    // Starting point
    D(1, 1) = C(0, 0);

    // Subsequence: first row can start anywhere
    if (subseq) {
        for (Eigen::Index j = 1; j <= m; ++j) {
            D(1, j) = C(0, j - 1);
        }
    }

    // Fill in the rest using standard DTW recurrence
    for (Eigen::Index i = 1; i <= n; ++i) {
        for (Eigen::Index j = 1; j <= m; ++j) {
            if (i == 1 && (subseq || j == 1)) continue;

            Real cost = C(i - 1, j - 1);
            Real val = std::min({
                D(i - 1, j - 1),  // diagonal
                D(i - 1, j),      // vertical
                D(i, j - 1)       // horizontal
            });

            D(i, j) = cost + val;
        }
    }

    // Return the core matrix (without padding)
    return D.block(1, 1, n, m);
}

std::pair<ArrayXXr, std::vector<std::pair<int, int>>> dtw_backtrack(
    const ArrayXXr& X,
    const ArrayXXr& Y,
    bool subseq,
    const std::string& metric) {

    // Compute cost matrix
    ArrayXXr C = cdist_for_metric(X, Y, metric);

    Eigen::Index n = C.rows();
    Eigen::Index m = C.cols();

    // Initialize accumulated cost matrix and step matrix
    ArrayXXr D = ArrayXXr::Constant(n + 1, m + 1, std::numeric_limits<Real>::infinity());
    Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> steps =
        Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(n + 1, m + 1);

    // 0 = diagonal, 1 = horizontal (j-1), 2 = vertical (i-1)
    D(1, 1) = C(0, 0);

    if (subseq) {
        for (Eigen::Index j = 1; j <= m; ++j) {
            D(1, j) = C(0, j - 1);
            steps(1, j) = 1;  // horizontal
        }
    }

    // Fill matrices
    for (Eigen::Index i = 1; i <= n; ++i) {
        for (Eigen::Index j = 1; j <= m; ++j) {
            if (i == 1 && (subseq || j == 1)) continue;

            Real cost = C(i - 1, j - 1);
            Real diag = D(i - 1, j - 1);
            Real vert = D(i - 1, j);
            Real horiz = D(i, j - 1);

            if (diag <= vert && diag <= horiz) {
                D(i, j) = cost + diag;
                steps(i, j) = 0;
            } else if (horiz <= vert) {
                D(i, j) = cost + horiz;
                steps(i, j) = 1;
            } else {
                D(i, j) = cost + vert;
                steps(i, j) = 2;
            }
        }
    }

    // Backtrack
    std::vector<std::pair<int, int>> path;

    int i = static_cast<int>(n);
    int j;

    if (subseq) {
        // Find best ending point in last row
        j = 1;
        for (int jj = 2; jj <= m; ++jj) {
            if (D(n, jj) < D(n, j)) {
                j = jj;
            }
        }
    } else {
        j = static_cast<int>(m);
    }

    // Backtrack
    while (i > 0 && j > 0) {
        path.push_back({i - 1, j - 1});  // Convert to 0-indexed

        int step = steps(i, j);
        if (step == 0) {
            --i;
            --j;
        } else if (step == 1) {
            --j;
        } else {
            --i;
        }

        // Stop at start for non-subsequence
        if (!subseq && i == 0 && j == 0) break;
        // Stop when reaching first row for subsequence
        if (subseq && i == 0) break;
    }

    // Reverse path (we built it backwards)
    std::reverse(path.begin(), path.end());

    ArrayXXr D_core = D.block(1, 1, n, m);
    return {D_core, path};
}

std::vector<std::pair<int, int>> dtw_backtracking(
    const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>& steps,
    bool subseq,
    std::optional<int> start) {

    Eigen::Index n = steps.rows();
    Eigen::Index m = steps.cols();

    int i = static_cast<int>(n) - 1;
    int j = start.value_or(static_cast<int>(m) - 1);

    std::vector<std::pair<int, int>> path;
    path.push_back({i, j});

    while ((subseq && i > 0) || (!subseq && (i > 0 || j > 0))) {
        int step = steps(i, j);

        // Default steps: 0 = diagonal, 1 = horizontal, 2 = vertical
        if (step == 0) {
            --i;
            --j;
        } else if (step == 1) {
            --j;
        } else {
            --i;
        }

        if (i < 0 || j < 0) break;

        path.push_back({i, j});
    }

    std::reverse(path.begin(), path.end());
    return path;
}

} // namespace sequence
} // namespace librosa
