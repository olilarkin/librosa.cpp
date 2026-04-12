#include <librosa/core/pitch.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/sequence.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include <internal/yin_utils.hpp>
#include <incbeta/incbeta.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>

namespace librosa {

namespace {

// Compute gradient along axis
ArrayXXr gradient(const ArrayXXr& S, int axis) {
    ArrayXXr grad = ArrayXXr::Zero(S.rows(), S.cols());

    if (axis == 0 || axis == -2) {
        // Gradient along rows
        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            // Forward difference at start
            grad(0, j) = S(1, j) - S(0, j);
            // Backward difference at end
            grad(S.rows() - 1, j) = S(S.rows() - 1, j) - S(S.rows() - 2, j);
            // Central difference in middle
            for (Eigen::Index i = 1; i < S.rows() - 1; ++i) {
                grad(i, j) = (S(i + 1, j) - S(i - 1, j)) / 2;
            }
        }
    } else {
        // Gradient along columns
        for (Eigen::Index i = 0; i < S.rows(); ++i) {
            grad(i, 0) = S(i, 1) - S(i, 0);
            grad(i, S.cols() - 1) = S(i, S.cols() - 1) - S(i, S.cols() - 2);
            for (Eigen::Index j = 1; j < S.cols() - 1; ++j) {
                grad(i, j) = (S(i, j + 1) - S(i, j - 1)) / 2;
            }
        }
    }

    return grad;
}

} // anonymous namespace

using internal::check_yin_params;
using internal::cumulative_mean_normalized_difference;
using internal::parabolic_interpolation;

// ============================================================================
// Pitch Tuning
// ============================================================================

Real pitch_tuning(const ArrayXr& frequencies, Real resolution, int bins_per_octave) {
    // Filter out DC components
    std::vector<Real> valid_freqs;
    for (Eigen::Index i = 0; i < frequencies.size(); ++i) {
        if (frequencies(i) > 0) {
            valid_freqs.push_back(frequencies(i));
        }
    }

    if (valid_freqs.empty()) {
        return 0.0;
    }

    // Compute residual relative to number of bins
    std::vector<Real> residuals;
    for (Real freq : valid_freqs) {
        Real oct = hz_to_octs(freq);
        Real residual = std::fmod(bins_per_octave * oct, 1.0);

        // Adjust for wrong side of semitone
        if (residual >= 0.5) {
            residual -= 1.0;
        }

        residuals.push_back(residual);
    }

    // Create histogram
    int n_bins = static_cast<int>(std::ceil(1.0 / resolution)) + 1;
    std::vector<int> counts(n_bins - 1, 0);

    for (Real r : residuals) {
        // Map residual [-0.5, 0.5) to bin index [0, n_bins-1)
        int bin = static_cast<int>((r + 0.5) / resolution);
        bin = std::max(0, std::min(bin, n_bins - 2));
        counts[bin]++;
    }

    // Find histogram peak
    int max_idx = 0;
    int max_count = counts[0];
    for (size_t i = 1; i < counts.size(); ++i) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            max_idx = static_cast<int>(i);
        }
    }

    // Convert bin index back to tuning
    Real tuning = -0.5 + max_idx * resolution;
    return tuning;
}

// ============================================================================
// Estimate Tuning
// ============================================================================

Real estimate_tuning(const ArrayXr& y, Real sr, int n_fft,
                     Real resolution, int bins_per_octave,
                     Real fmin, Real fmax, Real threshold) {
    auto [pitch, mag] = piptrack(y, sr, n_fft, std::nullopt, fmin, fmax, threshold);

    // Only count magnitude where frequency > 0
    std::vector<Real> valid_pitches;

    // Find median magnitude where pitch > 0
    std::vector<Real> valid_mags;
    for (Eigen::Index i = 0; i < pitch.size(); ++i) {
        if (pitch.data()[i] > 0) {
            valid_mags.push_back(mag.data()[i]);
        }
    }

    Real mag_threshold = 0.0;
    if (!valid_mags.empty()) {
        std::sort(valid_mags.begin(), valid_mags.end());
        mag_threshold = valid_mags[valid_mags.size() / 2];  // Median
    }

    // Collect pitches above threshold
    for (Eigen::Index i = 0; i < pitch.size(); ++i) {
        if (pitch.data()[i] > 0 && mag.data()[i] >= mag_threshold) {
            valid_pitches.push_back(pitch.data()[i]);
        }
    }

    if (valid_pitches.empty()) {
        return 0.0;
    }

    ArrayXr pitches = Eigen::Map<ArrayXr>(valid_pitches.data(), valid_pitches.size());
    return pitch_tuning(pitches, resolution, bins_per_octave);
}

Real estimate_tuning(const ArrayXXr& S, Real sr, int n_fft,
                     Real resolution, int bins_per_octave,
                     Real fmin, Real fmax, Real threshold) {
    auto [pitch, mag] = piptrack(S, sr, n_fft, std::nullopt, fmin, fmax, threshold);

    std::vector<Real> valid_mags;
    for (Eigen::Index i = 0; i < pitch.size(); ++i) {
        if (pitch.data()[i] > 0) {
            valid_mags.push_back(mag.data()[i]);
        }
    }

    Real mag_threshold = 0.0;
    if (!valid_mags.empty()) {
        std::sort(valid_mags.begin(), valid_mags.end());
        mag_threshold = valid_mags[valid_mags.size() / 2];
    }

    std::vector<Real> valid_pitches;
    for (Eigen::Index i = 0; i < pitch.size(); ++i) {
        if (pitch.data()[i] > 0 && mag.data()[i] >= mag_threshold) {
            valid_pitches.push_back(pitch.data()[i]);
        }
    }

    if (valid_pitches.empty()) {
        return 0.0;
    }

    ArrayXr pitches = Eigen::Map<ArrayXr>(valid_pitches.data(), valid_pitches.size());
    return pitch_tuning(pitches, resolution, bins_per_octave);
}

// ============================================================================
// Piptrack
// ============================================================================

std::pair<ArrayXXr, ArrayXXr> piptrack(
    const ArrayXr& y, Real sr, int n_fft,
    std::optional<int> hop_length_opt,
    Real fmin, Real fmax, Real threshold, bool center) {

    int hop_length = hop_length_opt.value_or(n_fft / 4);

    // Compute STFT
    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt, WindowType::Hann, center);
    ArrayXXr S = magnitude(D);

    return piptrack(S, sr, n_fft, hop_length, fmin, fmax, threshold);
}

std::pair<ArrayXXr, ArrayXXr> piptrack(
    const ArrayXXr& S, Real sr, int n_fft,
    std::optional<int> hop_length_opt,
    Real fmin, Real fmax, Real threshold) {

    // Ensure valid frequency range
    fmin = std::max(fmin, Real(0.0));
    fmax = std::min(fmax, sr / 2);

    ArrayXr fft_freqs = fft_frequencies(sr, n_fft);

    // Compute gradient and parabolic interpolation
    ArrayXXr avg = gradient(S, -2);
    ArrayXXr shift = parabolic_interpolation(S);
    ArrayXXr dskew = 0.5 * avg * shift;

    // Pre-allocate output
    ArrayXXr pitches = ArrayXXr::Zero(S.rows(), S.cols());
    ArrayXXr mags = ArrayXXr::Zero(S.rows(), S.cols());

    // For each frame, find local maxima above threshold
    for (Eigen::Index t = 0; t < S.cols(); ++t) {
        // Find reference value (max in column)
        Real ref_value = threshold * S.col(t).maxCoeff();

        // Find local maxima
        ArrayXr col = S.col(t);
        auto local_max = util::localmax(col);

        for (Eigen::Index f = 0; f < S.rows(); ++f) {
            // Check frequency range
            if (fft_freqs(f) < fmin || fft_freqs(f) >= fmax) {
                continue;
            }

            // Check local maximum and threshold
            if (local_max(f) && S(f, t) > ref_value) {
                pitches(f, t) = (f + shift(f, t)) * sr / n_fft;
                mags(f, t) = S(f, t) + dskew(f, t);
            }
        }
    }

    return {pitches, mags};
}

// ============================================================================
// YIN
// ============================================================================

ArrayXr yin(const ArrayXr& y, Real fmin, Real fmax, Real sr,
            int frame_length, std::optional<int> hop_length_opt,
            Real trough_threshold, bool center, PadMode pad_mode) {

    if (fmin <= 0 || fmax <= 0) {
        throw ParameterError("fmin and fmax must be provided");
    }

    check_yin_params(sr, fmax, fmin, frame_length);

    int hop_length = hop_length_opt.value_or(frame_length / 4);

    // Validate audio
    util::valid_audio(y);

    // Pad if centering
    ArrayXr y_padded;
    if (center) {
        int pad_size = frame_length / 2;
        y_padded.resize(y.size() + 2 * pad_size);
        y_padded.head(pad_size).setZero();
        y_padded.segment(pad_size, y.size()) = y;
        y_padded.tail(pad_size).setZero();
    } else {
        y_padded = y;
    }

    // Frame the audio
    ArrayXXr y_frames = util::frame(y_padded, frame_length, hop_length);

    // Calculate periods
    int min_period = static_cast<int>(std::floor(sr / fmax));
    int max_period = std::min(static_cast<int>(std::ceil(sr / fmin)), frame_length - 1);

    // Compute cumulative mean normalized difference
    ArrayXXr yin_frames = cumulative_mean_normalized_difference(y_frames, min_period, max_period);

    // Parabolic interpolation
    ArrayXXr parabolic_shifts = parabolic_interpolation(yin_frames);

    // Find local minima
    Eigen::Index n_frames = yin_frames.cols();
    Eigen::Index n_periods = yin_frames.rows();

    ArrayXr f0(n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        // Find troughs (local minima)
        Eigen::Array<bool, Eigen::Dynamic, 1> is_trough(n_periods);
        is_trough(0) = yin_frames(0, t) < yin_frames(1, t);
        for (Eigen::Index i = 1; i < n_periods - 1; ++i) {
            is_trough(i) = (yin_frames(i, t) < yin_frames(i - 1, t)) &&
                           (yin_frames(i, t) <= yin_frames(i + 1, t));
        }
        is_trough(n_periods - 1) = false;

        // Find first trough below threshold
        Eigen::Index yin_period = -1;
        for (Eigen::Index i = 0; i < n_periods; ++i) {
            if (is_trough(i) && yin_frames(i, t) < trough_threshold) {
                yin_period = i;
                break;
            }
        }

        // If no trough below threshold, use global minimum
        if (yin_period < 0) {
            Eigen::Index min_idx;
            yin_frames.col(t).minCoeff(&min_idx);
            yin_period = min_idx;
        }

        // Refine with parabolic interpolation
        Real refined_period = min_period + yin_period + parabolic_shifts(yin_period, t);

        // Convert to frequency
        f0(t) = sr / refined_period;
    }

    return f0;
}

// ============================================================================
// pYIN
// ============================================================================

namespace {

// Boltzmann PMF: P(k | lambda, N)
Real boltzmann_pmf(int k, Real lambda, int N) {
    if (N <= 0) return 0.0;
    Real num = (1.0 - std::exp(-lambda)) * std::exp(-lambda * k);
    Real denom = 1.0 - std::exp(-lambda * N);
    if (std::abs(denom) < 1e-30) return 0.0;
    return num / denom;
}

// Kronecker product of two matrices
ArrayXXr kronecker(const ArrayXXr& A, const ArrayXXr& B) {
    Eigen::Index m = A.rows(), n = A.cols();
    Eigen::Index p = B.rows(), q = B.cols();
    ArrayXXr result(m * p, n * q);
    for (Eigen::Index i = 0; i < m; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            result.block(i * p, j * q, p, q) = A(i, j) * B;
        }
    }
    return result;
}

} // anonymous namespace

PyinResult pyin(
    const ArrayXr& y,
    Real fmin,
    Real fmax,
    Real sr,
    int frame_length,
    std::optional<int> hop_length_opt,
    int n_thresholds,
    Real beta_a,
    Real beta_b,
    Real boltzmann_parameter,
    Real resolution,
    Real max_transition_rate,
    Real switch_prob,
    Real no_trough_prob,
    std::optional<Real> fill_na,
    bool center,
    PadMode pad_mode) {

    if (fmin <= 0 || fmax <= 0) {
        throw ParameterError("fmin and fmax must be provided");
    }

    check_yin_params(sr, fmax, fmin, frame_length);

    int hop_length = hop_length_opt.value_or(frame_length / 4);

    // Validate audio
    util::valid_audio(y);

    // Pad if centering
    ArrayXr y_padded;
    if (center) {
        int pad_size = frame_length / 2;
        y_padded.resize(y.size() + 2 * pad_size);
        y_padded.head(pad_size).setZero();
        y_padded.segment(pad_size, y.size()) = y;
        y_padded.tail(pad_size).setZero();
    } else {
        y_padded = y;
    }

    // Frame audio
    ArrayXXr y_frames = util::frame(y_padded, frame_length, hop_length);

    // Calculate minimum and maximum periods
    int min_period = static_cast<int>(std::floor(sr / fmax));
    int max_period = std::min(static_cast<int>(std::ceil(sr / fmin)), frame_length - 1);

    // Cumulative mean normalized difference
    ArrayXXr yin_frames = cumulative_mean_normalized_difference(y_frames, min_period, max_period);

    // Parabolic interpolation
    ArrayXXr parabolic_shifts = parabolic_interpolation(yin_frames);

    Eigen::Index n_periods = yin_frames.rows();
    Eigen::Index n_frames = yin_frames.cols();

    // ---- Step 7: Build threshold prior (Beta distribution) ----
    ArrayXr thresholds = ArrayXr::LinSpaced(n_thresholds + 1, 0.0, 1.0);
    ArrayXr beta_cdf(n_thresholds + 1);
    for (int i = 0; i <= n_thresholds; ++i) {
        beta_cdf(i) = incbeta(beta_a, beta_b, thresholds(i));
    }
    // beta_probs = diff(beta_cdf)
    ArrayXr beta_probs(n_thresholds);
    for (int i = 0; i < n_thresholds; ++i) {
        beta_probs(i) = beta_cdf(i + 1) - beta_cdf(i);
    }

    // ---- Step 8: Pitch bin dimensions ----
    int n_bins_per_semitone = static_cast<int>(std::ceil(1.0 / resolution));
    int n_pitch_bins = static_cast<int>(std::floor(12.0 * n_bins_per_semitone * std::log2(fmax / fmin))) + 1;

    // ---- Step 9: Compute observation probabilities ----
    // observation_probs: (2 * n_pitch_bins, n_frames)
    ArrayXXr observation_probs = ArrayXXr::Zero(2 * n_pitch_bins, n_frames);
    ArrayXr voiced_prob_arr(n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        // Compute yin_probs for this frame (mirrors __pyin_helper)
        ArrayXr yin_frame = yin_frames.col(t);

        // Find troughs via localmin
        auto is_trough = util::localmin(yin_frame);
        // Special first-element rule
        if (n_periods >= 2) {
            is_trough(0) = yin_frame(0) < yin_frame(1);
        }

        // Collect trough indices
        std::vector<Eigen::Index> trough_index;
        for (Eigen::Index i = 0; i < n_periods; ++i) {
            if (is_trough(i)) {
                trough_index.push_back(i);
            }
        }

        if (trough_index.empty()) {
            // No troughs: all observation prob stays 0 for voiced states
            voiced_prob_arr(t) = 0.0;
            // Unvoiced observation = 1.0 / n_pitch_bins (uniform)
            for (int b = 0; b < n_pitch_bins; ++b) {
                observation_probs(n_pitch_bins + b, t) = 1.0 / n_pitch_bins;
            }
            continue;
        }

        int n_troughs = static_cast<int>(trough_index.size());

        // trough_heights
        ArrayXr trough_heights(n_troughs);
        for (int k = 0; k < n_troughs; ++k) {
            trough_heights(k) = yin_frame(trough_index[k]);
        }

        // trough_thresholds: (n_troughs x n_thresholds) boolean
        // trough_thresholds(k, j) = trough_heights(k) < thresholds(j+1)
        // Using Eigen bool array
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            trough_thresholds(n_troughs, n_thresholds);
        for (int k = 0; k < n_troughs; ++k) {
            for (int j = 0; j < n_thresholds; ++j) {
                trough_thresholds(k, j) = trough_heights(k) < thresholds(j + 1);
            }
        }

        // trough_positions = cumsum(trough_thresholds, axis=0) - 1
        // n_troughs_per_threshold = count_nonzero(trough_thresholds, axis=0)
        Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            trough_positions(n_troughs, n_thresholds);
        Eigen::Array<int, Eigen::Dynamic, 1> n_troughs_per_thresh(n_thresholds);

        for (int j = 0; j < n_thresholds; ++j) {
            int cumsum = 0;
            int count = 0;
            for (int k = 0; k < n_troughs; ++k) {
                if (trough_thresholds(k, j)) {
                    cumsum++;
                    count++;
                }
                trough_positions(k, j) = cumsum - 1;
            }
            n_troughs_per_thresh(j) = count;
        }

        // Compute trough_prior using Boltzmann PMF
        // trough_prior(k, j) = boltzmann_pmf(trough_positions(k,j), boltzmann_parameter, n_troughs_per_thresh(j))
        //   but only if trough_thresholds(k, j) is true, else 0
        ArrayXXr trough_prior(n_troughs, n_thresholds);
        for (int k = 0; k < n_troughs; ++k) {
            for (int j = 0; j < n_thresholds; ++j) {
                if (trough_thresholds(k, j)) {
                    trough_prior(k, j) = boltzmann_pmf(
                        trough_positions(k, j),
                        boltzmann_parameter,
                        n_troughs_per_thresh(j));
                } else {
                    trough_prior(k, j) = 0.0;
                }
            }
        }

        // probs = trough_prior.dot(beta_probs)  (matrix-vector product)
        // trough_prior: (n_troughs x n_thresholds), beta_probs: (n_thresholds,)
        // probs: (n_troughs,)
        ArrayXr probs(n_troughs);
        for (int k = 0; k < n_troughs; ++k) {
            Real sum = 0.0;
            for (int j = 0; j < n_thresholds; ++j) {
                sum += trough_prior(k, j) * beta_probs(j);
            }
            probs(k) = sum;
        }

        // Add no_trough_prob mass to global minimum
        Eigen::Index global_min_idx = 0;
        Real global_min_val = trough_heights(0);
        for (int k = 1; k < n_troughs; ++k) {
            if (trough_heights(k) < global_min_val) {
                global_min_val = trough_heights(k);
                global_min_idx = k;
            }
        }

        // Count thresholds below global minimum
        int n_thresholds_below_min = 0;
        for (int j = 0; j < n_thresholds; ++j) {
            if (!trough_thresholds(global_min_idx, j)) {
                n_thresholds_below_min++;
            }
        }
        Real extra_prob = 0.0;
        for (int j = 0; j < n_thresholds_below_min; ++j) {
            extra_prob += beta_probs(j);
        }
        probs(global_min_idx) += no_trough_prob * extra_prob;

        // Now map trough probs to pitch bins
        // For each trough with nonzero prob:
        for (int k = 0; k < n_troughs; ++k) {
            if (probs(k) <= 0.0) continue;

            Eigen::Index period_idx = trough_index[k];
            Real period = min_period + period_idx + parabolic_shifts(period_idx, t);
            Real f0_candidate = sr / period;

            // Map to pitch bin
            Real bin_float = 12.0 * n_bins_per_semitone * std::log2(f0_candidate / fmin);
            int bin_idx = static_cast<int>(std::round(bin_float));
            bin_idx = std::max(0, std::min(bin_idx, n_pitch_bins - 1));

            // Accumulate (multiple troughs can map to same bin)
            observation_probs(bin_idx, t) += probs(k);
        }

        // voiced_prob = clip(sum(obs_probs[:n_pitch_bins, t]), 0, 1)
        Real vp = 0.0;
        for (int b = 0; b < n_pitch_bins; ++b) {
            vp += observation_probs(b, t);
        }
        vp = std::max(0.0, std::min(1.0, vp));
        voiced_prob_arr(t) = vp;

        // Unvoiced observation = (1 - voiced_prob) / n_pitch_bins
        Real unvoiced_obs = (1.0 - vp) / n_pitch_bins;
        for (int b = 0; b < n_pitch_bins; ++b) {
            observation_probs(n_pitch_bins + b, t) = unvoiced_obs;
        }
    }

    // ---- Step 10: Build transition matrix ----
    int max_semitones_per_frame = static_cast<int>(
        std::round(max_transition_rate * 12.0 * hop_length / sr));
    int transition_width = max_semitones_per_frame * n_bins_per_semitone + 1;

    ArrayXXr transition = sequence::transition_local(
        n_pitch_bins, transition_width, WindowType::Triangle, false);

    ArrayXXr t_switch = sequence::transition_loop(2, 1.0 - switch_prob);

    ArrayXXr full_transition = kronecker(t_switch, transition);

    // ---- Step 11: Viterbi decode ----
    ArrayXr p_init = ArrayXr::Constant(2 * n_pitch_bins, 1.0 / (2 * n_pitch_bins));
    std::vector<int> states = sequence::viterbi(observation_probs, full_transition, p_init);

    // ---- Step 12: Extract results ----
    // Build frequency lookup table
    ArrayXr freqs(n_pitch_bins);
    for (int i = 0; i < n_pitch_bins; ++i) {
        freqs(i) = fmin * std::pow(2.0, static_cast<Real>(i) / (12.0 * n_bins_per_semitone));
    }

    PyinResult result;
    result.f0.resize(n_frames);
    result.voiced_flag.resize(n_frames);
    result.voiced_prob.resize(n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        int state = states[t];
        int pitch_bin = state % n_pitch_bins;
        result.f0(t) = freqs(pitch_bin);
        result.voiced_flag(t) = state < n_pitch_bins;
        result.voiced_prob(t) = voiced_prob_arr(t);
    }

    // Apply fill_na for unvoiced frames
    if (fill_na.has_value()) {
        Real fill_val = fill_na.value();
        for (Eigen::Index t = 0; t < n_frames; ++t) {
            if (!result.voiced_flag(t)) {
                result.f0(t) = fill_val;
            }
        }
    }

    return result;
}

} // namespace librosa
