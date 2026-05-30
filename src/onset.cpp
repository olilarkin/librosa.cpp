#include <librosa/onset.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/feature/spectral.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace librosa {
namespace onset {

// ============================================================================
// Maximum Filter 1D
// ============================================================================

ArrayXXr maximum_filter1d(const ArrayXXr& S, int size, int axis) {
    if (size < 1) {
        throw ParameterError("max_size must be a positive integer");
    }

    if (size == 1) {
        return S;  // No filtering needed
    }

    ArrayXXr result(S.rows(), S.cols());
    int half_size = size / 2;

    if (axis == -2 || axis == 0) {
        // Filter along rows (frequency axis for spectrograms)
        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            for (Eigen::Index i = 0; i < S.rows(); ++i) {
                Real max_val = S(i, j);
                for (int k = -half_size; k <= half_size; ++k) {
                    Eigen::Index idx = i + k;
                    if (idx >= 0 && idx < S.rows()) {
                        max_val = std::max(max_val, S(idx, j));
                    }
                }
                result(i, j) = max_val;
            }
        }
    } else if (axis == -1 || axis == 1) {
        // Filter along columns (time axis)
        for (Eigen::Index i = 0; i < S.rows(); ++i) {
            for (Eigen::Index j = 0; j < S.cols(); ++j) {
                Real max_val = S(i, j);
                for (int k = -half_size; k <= half_size; ++k) {
                    Eigen::Index idx = j + k;
                    if (idx >= 0 && idx < S.cols()) {
                        max_val = std::max(max_val, S(i, idx));
                    }
                }
                result(i, j) = max_val;
            }
        }
    } else {
        throw ParameterError("Invalid axis for maximum_filter1d");
    }

    return result;
}

// ============================================================================
// Match Events
// ============================================================================

std::vector<Eigen::Index> match_events(
    const std::vector<Eigen::Index>& events_from,
    const std::vector<Eigen::Index>& events_to,
    bool left,
    bool right) {

    if (events_from.empty() || events_to.empty()) {
        throw ParameterError("Attempting to match empty event list");
    }

    // Sort events_to for binary search
    std::vector<Eigen::Index> sorted_to = events_to;
    std::sort(sorted_to.begin(), sorted_to.end());

    std::vector<Eigen::Index> result(events_from.size());

    for (size_t i = 0; i < events_from.size(); ++i) {
        Eigen::Index event = events_from[i];

        // Binary search to find insertion point
        auto it = std::lower_bound(sorted_to.begin(), sorted_to.end(), event);
        size_t pos = it - sorted_to.begin();

        // Find best match considering left/right constraints
        Eigen::Index best_idx = 0;
        Eigen::Index best_diff = std::numeric_limits<Eigen::Index>::max();

        // Check current position
        if (pos < sorted_to.size()) {
            Eigen::Index diff = std::abs(sorted_to[pos] - event);
            if (right || sorted_to[pos] <= event) {
                if (diff < best_diff) {
                    best_diff = diff;
                    best_idx = pos;
                }
            }
        }

        // Check position to the left
        if (pos > 0 && left) {
            Eigen::Index diff = std::abs(sorted_to[pos - 1] - event);
            if (diff < best_diff) {
                best_diff = diff;
                best_idx = pos - 1;
            }
        }

        // Check position to the right (if not already checked)
        if (pos < sorted_to.size() - 1 && right) {
            Eigen::Index diff = std::abs(sorted_to[pos + 1] - event);
            if (diff < best_diff) {
                best_diff = diff;
                best_idx = pos + 1;
            }
        }

        // Find the original index in events_to
        Eigen::Index matched_value = sorted_to[best_idx];
        auto orig_it = std::find(events_to.begin(), events_to.end(), matched_value);
        result[i] = orig_it - events_to.begin();
    }

    return result;
}

// ============================================================================
// Simple IIR Filter for Detrending
// ============================================================================

namespace {

ArrayXr lfilter_detrend(const ArrayXr& x) {
    // Implements y[n] = x[n] - x[n-1] + 0.99 * y[n-1]
    // This is a simple high-pass filter to remove DC component
    ArrayXr y(x.size());

    if (x.size() == 0) return y;

    y(0) = x(0);
    for (Eigen::Index n = 1; n < x.size(); ++n) {
        y(n) = x(n) - x(n - 1) + 0.99 * y(n - 1);
    }

    return y;
}

Real median_of_column(const ArrayXXr& data, Eigen::Index col) {
    std::vector<Real> values(static_cast<size_t>(data.rows()));
    for (Eigen::Index row = 0; row < data.rows(); ++row) {
        values[static_cast<size_t>(row)] = data(row, col);
    }

    size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    Real upper = values[mid];

    if (values.size() % 2 == 1) {
        return upper;
    }

    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    return 0.5 * (values[mid - 1] + upper);
}

ArrayXr aggregate_onset_env(const ArrayXXr& onset_env, AggregateFunc aggregate) {
    if (aggregate == AggregateFunc::Mean) {
        return onset_env.colwise().mean();
    }

    if (aggregate == AggregateFunc::Median) {
        ArrayXr env(onset_env.cols());
        for (Eigen::Index col = 0; col < onset_env.cols(); ++col) {
            env(col) = median_of_column(onset_env, col);
        }
        return env;
    }

    throw ParameterError("Unsupported onset strength aggregate");
}

} // anonymous namespace

// ============================================================================
// Onset Strength
// ============================================================================

ArrayXr onset_strength(
    const ArrayXr& y,
    Real sr,
    int n_fft,
    int hop_length,
    int lag,
    int max_size,
    bool detrend,
    bool center,
    AggregateFunc aggregate) {

    // Compute mel spectrogram
    ArrayXXr S = feature::melspectrogram(y, sr, n_fft, hop_length);

    // Convert to dB
    S = power_to_db(S, 1.0, 1e-10, 80.0);

    return onset_strength(S, sr, n_fft, hop_length, lag, max_size, detrend,
                          center, aggregate);
}

ArrayXr onset_strength(
    const ArrayXXr& S,
    Real sr,
    int n_fft,
    int hop_length,
    int lag,
    int max_size,
    bool detrend,
    bool center,
    AggregateFunc aggregate) {

    if (!util::is_positive_int(lag)) {
        throw ParameterError("lag must be a positive integer");
    }
    if (!util::is_positive_int(max_size)) {
        throw ParameterError("max_size must be a positive integer");
    }

    // Compute reference spectrogram with max filter for vibrato suppression
    ArrayXXr ref;
    if (max_size == 1) {
        ref = S;
    } else {
        ref = maximum_filter1d(S, max_size, -2);  // Along frequency axis
    }

    // Compute difference to the reference, spaced by lag
    // onset_env = S[..., lag:] - ref[..., :-lag]
    Eigen::Index n_frames = S.cols();
    if (n_frames <= lag) {
        return ArrayXr::Zero(n_frames);
    }

    ArrayXXr onset_env = S.block(0, lag, S.rows(), n_frames - lag) -
                         ref.block(0, 0, ref.rows(), n_frames - lag);

    // Discard negatives (decreasing amplitude)
    onset_env = onset_env.max(0.0);

    // Aggregate across frequency
    ArrayXr env = aggregate_onset_env(onset_env, aggregate);

    // Compensate for lag and centering
    int pad_width = lag;
    if (center) {
        pad_width += n_fft / (2 * hop_length);
    }

    // Pad the beginning
    ArrayXr padded(env.size() + pad_width);
    padded.head(pad_width).setZero();
    padded.tail(env.size()) = env;

    // Remove DC component if requested
    if (detrend) {
        padded = lfilter_detrend(padded);
    }

    // Trim to match input duration
    if (center) {
        Eigen::Index target_size = S.cols();
        if (padded.size() > target_size) {
            padded.conservativeResize(target_size);
        }
    }

    return padded;
}

// ============================================================================
// Onset Strength Multi
// ============================================================================

ArrayXXr onset_strength_multi(
    const ArrayXr& y,
    Real sr,
    int n_fft,
    int hop_length,
    int lag,
    int max_size,
    bool detrend,
    bool center,
    const std::vector<int>& channels) {

    // Compute mel spectrogram
    ArrayXXr S = feature::melspectrogram(y, sr, n_fft, hop_length);

    // Convert to dB
    S = power_to_db(S, 1.0, 1e-10, 80.0);

    return onset_strength_multi(S, sr, n_fft, hop_length, lag, max_size, detrend, center, channels);
}

ArrayXXr onset_strength_multi(
    const ArrayXXr& S,
    Real sr,
    int n_fft,
    int hop_length,
    int lag,
    int max_size,
    bool detrend,
    bool center,
    const std::vector<int>& channels) {

    if (!util::is_positive_int(lag)) {
        throw ParameterError("lag must be a positive integer");
    }
    if (!util::is_positive_int(max_size)) {
        throw ParameterError("max_size must be a positive integer");
    }

    // Compute reference spectrogram with max filter for vibrato suppression
    ArrayXXr ref;
    if (max_size == 1) {
        ref = S;
    } else {
        ref = maximum_filter1d(S, max_size, -2);
    }

    // Compute difference to the reference, spaced by lag
    Eigen::Index n_frames = S.cols();
    Eigen::Index n_freq = S.rows();
    if (n_frames <= lag) {
        if (channels.empty()) {
            return ArrayXXr::Zero(1, n_frames);
        } else {
            return ArrayXXr::Zero(static_cast<Eigen::Index>(channels.size()) - 1, n_frames);
        }
    }

    ArrayXXr onset_env = S.block(0, lag, n_freq, n_frames - lag) -
                         ref.block(0, 0, n_freq, n_frames - lag);

    // Discard negatives
    onset_env = onset_env.max(0.0);

    // Aggregate within channels
    ArrayXXr aggregated;
    if (channels.empty()) {
        // Single channel: mean across all frequencies
        aggregated.resize(1, onset_env.cols());
        aggregated.row(0) = onset_env.colwise().mean();
    } else {
        // Multiple channels: mean within each sub-band
        Eigen::Index n_channels = static_cast<Eigen::Index>(channels.size()) - 1;
        aggregated.resize(n_channels, onset_env.cols());
        for (Eigen::Index c = 0; c < n_channels; ++c) {
            int start = channels[c];
            int end = channels[c + 1];
            int band_size = end - start;
            if (band_size <= 0 || start < 0 || end > onset_env.rows()) {
                throw ParameterError("Invalid channel boundaries");
            }
            aggregated.row(c) = onset_env.block(start, 0, band_size, onset_env.cols()).colwise().mean();
        }
    }

    // Compensate for lag and centering
    int pad_width = lag;
    if (center) {
        pad_width += n_fft / (2 * hop_length);
    }

    // Pad the beginning of each channel
    Eigen::Index n_channels = aggregated.rows();
    Eigen::Index env_cols = aggregated.cols();
    ArrayXXr padded(n_channels, env_cols + pad_width);
    padded.block(0, 0, n_channels, pad_width).setZero();
    padded.block(0, pad_width, n_channels, env_cols) = aggregated;

    // Remove DC component if requested
    if (detrend) {
        for (Eigen::Index c = 0; c < n_channels; ++c) {
            ArrayXr row = padded.row(c);
            padded.row(c) = lfilter_detrend(row);
        }
    }

    // Trim to match input duration
    if (center) {
        Eigen::Index target_size = S.cols();
        if (padded.cols() > target_size) {
            padded.conservativeResize(Eigen::NoChange, target_size);
        }
    }

    return padded;
}

// ============================================================================
// Onset Detection
// ============================================================================

std::vector<Eigen::Index> onset_detect(
    const ArrayXr& y,
    Real sr,
    int hop_length,
    bool backtrack,
    OnsetUnits units,
    bool normalize,
    int pre_max,
    int post_max,
    int pre_avg,
    int post_avg,
    Real delta,
    int wait) {

    // Compute onset envelope
    ArrayXr env = onset_strength(y, sr, 2048, hop_length);

    return onset_detect_envelope(env, sr, hop_length, backtrack, units, normalize,
                                  pre_max, post_max, pre_avg, post_avg, delta, wait);
}

std::vector<Eigen::Index> onset_detect_envelope(
    const ArrayXr& onset_envelope,
    Real sr,
    int hop_length,
    bool backtrack,
    OnsetUnits units,
    bool normalize,
    int pre_max,
    int post_max,
    int pre_avg,
    int post_avg,
    Real delta,
    int wait) {

    ArrayXr env = onset_envelope;

    // Normalize to [0, 1]
    if (normalize) {
        Real min_val = env.minCoeff();
        env = env - min_val;
        Real max_val = env.maxCoeff();
        if (max_val > util::tiny<Real>()) {
            env = env / max_val;
        }
    }

    // Check for valid onset envelope
    if (env.size() == 0 || !env.allFinite()) {
        return {};
    }

    // Match Python librosa: empty or all-zero envelopes produce no detections.
    if (env.abs().maxCoeff() <= util::tiny<Real>()) {
        return {};
    }

    // Set default parameters based on sample rate and hop length
    // These are the optimized values from librosa
    if (pre_max == 0) {
        pre_max = static_cast<int>(0.03 * sr / hop_length);  // 30ms
    }
    if (post_max == 0) {
        post_max = 1;  // 0ms
    }
    if (pre_avg == 0) {
        pre_avg = static_cast<int>(0.10 * sr / hop_length);  // 100ms
    }
    if (post_avg == 0) {
        post_avg = static_cast<int>(0.10 * sr / hop_length) + 1;  // 100ms
    }
    if (wait == 0) {
        wait = static_cast<int>(0.03 * sr / hop_length);  // 30ms
    }

    // Peak pick
    std::vector<Eigen::Index> onsets = util::peak_pick(
        env, pre_max, post_max, pre_avg, post_avg, delta, wait);

    // Backtrack if requested
    if (backtrack && !onsets.empty()) {
        onsets = onset_backtrack(onsets, env);
    }

    // Convert units
    if (units == OnsetUnits::Samples) {
        for (auto& onset : onsets) {
            onset = frames_to_samples(onset, hop_length);
        }
    } else if (units == OnsetUnits::Time) {
        throw ParameterError(
            "OnsetUnits::Time is not supported by onset_detect/onset_detect_envelope; "
            "use onset_detect_times for real-valued times.");
    }

    return onsets;
}

ArrayXr onset_detect_times(
    const ArrayXr& y,
    Real sr,
    int hop_length,
    bool backtrack,
    bool normalize) {

    std::vector<Eigen::Index> frames = onset_detect(
        y, sr, hop_length, backtrack, OnsetUnits::Frames, normalize);

    ArrayXr times(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        times(i) = frames_to_time(frames[i], sr, hop_length);
    }

    return times;
}

// ============================================================================
// Onset Backtracking
// ============================================================================

std::vector<Eigen::Index> onset_backtrack(
    const std::vector<Eigen::Index>& events,
    const ArrayXr& energy) {

    if (events.empty()) {
        return {};
    }

    if (energy.size() < 3) {
        return events;  // Not enough data for backtracking
    }

    // Find local minima
    // A minimum is where: energy[i] <= energy[i-1] && energy[i] < energy[i+1]
    std::vector<Eigen::Index> minima;
    minima.push_back(0);  // Pad with 0

    for (Eigen::Index i = 1; i < energy.size() - 1; ++i) {
        if (energy(i) <= energy(i - 1) && energy(i) < energy(i + 1)) {
            minima.push_back(i);
        }
    }

    if (minima.empty()) {
        return events;
    }

    // Match events to minima, only looking left
    std::vector<Eigen::Index> result = match_events(events, minima, true, false);

    // Convert indices back to actual minima positions
    std::vector<Eigen::Index> backtracked(events.size());
    for (size_t i = 0; i < events.size(); ++i) {
        backtracked[i] = minima[result[i]];
    }

    return backtracked;
}

} // namespace onset
} // namespace librosa
