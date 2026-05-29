#include <librosa/beat.hpp>
#include <librosa/onset.hpp>
#include <librosa/feature/rhythm.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include <librosa/filters.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace librosa {
namespace beat {

// ============================================================================
// Tempogram Implementation
// ============================================================================

ArrayXXr tempogram(
    const ArrayXr& onset_envelope,
    Real sr,
    int hop_length,
    int win_length,
    bool center,
    WindowType window,
    std::optional<Real> norm) {

    if (win_length < 1) {
        throw ParameterError("win_length must be a positive integer");
    }

    // Get window
    ArrayXr ac_window = get_window(window, win_length, true);

    Eigen::Index n = onset_envelope.size();

    // Pad if centering
    ArrayXr envelope = onset_envelope;
    if (center) {
        int pad_size = win_length / 2;
        ArrayXr padded(n + 2 * pad_size);

        // Linear ramp at start
        for (int i = 0; i < pad_size; ++i) {
            padded(i) = envelope(0) * static_cast<Real>(i) / pad_size;
        }

        // Copy data
        padded.segment(pad_size, n) = envelope;

        // Linear ramp at end
        for (int i = 0; i < pad_size; ++i) {
            padded(n + pad_size + i) = envelope(n - 1) * static_cast<Real>(pad_size - i) / pad_size;
        }

        envelope = padded;
    }

    // Frame the onset envelope
    Eigen::Index n_frames = center ? n : envelope.size() - win_length + 1;
    ArrayXXr odf_frame(win_length, n_frames);

    for (Eigen::Index i = 0; i < n_frames; ++i) {
        odf_frame.col(i) = envelope.segment(i, win_length);
    }

    // Window each frame
    for (Eigen::Index i = 0; i < n_frames; ++i) {
        odf_frame.col(i) *= ac_window;
    }

    // Autocorrelate each frame
    ArrayXXr tgram(win_length, n_frames);
    for (Eigen::Index i = 0; i < n_frames; ++i) {
        ArrayXr frame = odf_frame.col(i);
        ArrayXr ac = autocorrelate(frame, win_length);
        tgram.col(i) = ac;
    }

    // Normalize
    if (norm.has_value()) {
        tgram = util::normalize(tgram, norm.value(), 0);
    }

    return tgram;
}

ArrayXXr tempogram_audio(
    const ArrayXr& y,
    Real sr,
    int hop_length,
    int win_length,
    bool center,
    WindowType window,
    std::optional<Real> norm) {

    // Compute onset envelope
    ArrayXr envelope = onset::onset_strength(y, sr, 2048, hop_length);

    return tempogram(envelope, sr, hop_length, win_length, center, window, norm);
}

// ============================================================================
// Tempo Estimation Implementation
// ============================================================================

namespace {

// Compute tempo frequencies for tempogram
ArrayXr tempo_frequencies_internal(int win_length, Real sr, int hop_length) {
    ArrayXr freqs(win_length);

    Real frame_rate = sr / hop_length;

    for (int i = 0; i < win_length; ++i) {
        if (i == 0) {
            freqs(i) = 0.0;  // DC component
        } else {
            // BPM = 60 * frame_rate / lag
            freqs(i) = 60.0 * frame_rate / i;
        }
    }

    return freqs;
}

} // anonymous namespace

Real tempo(
    const ArrayXr& onset_envelope,
    Real sr,
    int hop_length,
    Real start_bpm,
    Real std_bpm,
    Real ac_size,
    std::optional<Real> max_tempo,
    bool aggregate) {

    if (start_bpm <= 0) {
        throw ParameterError("start_bpm must be strictly positive");
    }

    // Compute window length from ac_size
    int win_length = static_cast<int>(time_to_frames(ac_size, sr, hop_length));
    if (win_length < 1) win_length = 1;

    // Compute tempogram
    ArrayXXr tg = tempogram(onset_envelope, sr, hop_length, win_length);

    // Aggregate if requested
    ArrayXr tg_agg;
    if (aggregate) {
        tg_agg = tg.rowwise().mean();
    } else {
        // Return first frame for simplicity (full dynamic tempo would need different return type)
        tg_agg = tg.col(0);
    }

    // Get BPM values for each bin
    ArrayXr bpms = tempo_frequencies_internal(win_length, sr, hop_length);

    // Compute log-normal-like prior
    ArrayXr logprior(win_length);
    for (int i = 0; i < win_length; ++i) {
        if (bpms(i) > 0) {
            Real log_ratio = (std::log2(bpms(i)) - std::log2(start_bpm)) / std_bpm;
            logprior(i) = -0.5 * log_ratio * log_ratio;
        } else {
            logprior(i) = -std::numeric_limits<Real>::infinity();
        }
    }

    // Kill everything above max tempo
    if (max_tempo.has_value()) {
        for (int i = 0; i < win_length; ++i) {
            if (bpms(i) > max_tempo.value()) {
                logprior(i) = -std::numeric_limits<Real>::infinity();
            }
        }
    }

    // Find best period using log1p weighting
    ArrayXr weighted(win_length);
    for (int i = 0; i < win_length; ++i) {
        weighted(i) = std::log1p(1e6 * tg_agg(i)) + logprior(i);
    }

    Eigen::Index best_idx;
    weighted.maxCoeff(&best_idx);

    return bpms(best_idx);
}

Real tempo_audio(
    const ArrayXr& y,
    Real sr,
    int hop_length,
    Real start_bpm,
    Real std_bpm,
    Real ac_size,
    std::optional<Real> max_tempo,
    bool aggregate) {

    ArrayXr envelope = onset::onset_strength(y, sr, 2048, hop_length);
    return tempo(envelope, sr, hop_length, start_bpm, std_bpm, ac_size, max_tempo, aggregate);
}

ArrayXr tempo_frames(
    const ArrayXr& onset_envelope,
    Real sr,
    int hop_length,
    Real start_bpm,
    Real std_bpm,
    Real ac_size,
    std::optional<Real> max_tempo) {

    if (start_bpm <= 0) {
        throw ParameterError("start_bpm must be strictly positive");
    }

    int win_length = static_cast<int>(time_to_frames(ac_size, sr, hop_length));
    if (win_length < 1) win_length = 1;

    ArrayXXr tg = tempogram(onset_envelope, sr, hop_length, win_length);
    ArrayXr bpms = tempo_frequencies_internal(win_length, sr, hop_length);

    // Compute log-prior
    ArrayXr logprior(win_length);
    for (int i = 0; i < win_length; ++i) {
        if (bpms(i) > 0) {
            Real log_ratio = (std::log2(bpms(i)) - std::log2(start_bpm)) / std_bpm;
            logprior(i) = -0.5 * log_ratio * log_ratio;
        } else {
            logprior(i) = -std::numeric_limits<Real>::infinity();
        }
    }

    if (max_tempo.has_value()) {
        for (int i = 0; i < win_length; ++i) {
            if (bpms(i) > max_tempo.value()) {
                logprior(i) = -std::numeric_limits<Real>::infinity();
            }
        }
    }

    // Per-frame tempo estimation
    ArrayXr tempos(tg.cols());
    for (Eigen::Index f = 0; f < tg.cols(); ++f) {
        ArrayXr weighted(win_length);
        for (int i = 0; i < win_length; ++i) {
            weighted(i) = std::log1p(1e6 * tg(i, f)) + logprior(i);
        }
        Eigen::Index best_idx;
        weighted.maxCoeff(&best_idx);
        tempos(f) = bpms(best_idx);
    }

    return tempos;
}

// ============================================================================
// Beat Tracking Implementation
// ============================================================================

namespace {

// Normalize onsets by standard deviation
ArrayXr normalize_onsets(const ArrayXr& onsets) {
    if (onsets.size() <= 1) {
        return onsets / util::tiny<Real>();
    }

    Real mean = onsets.mean();
    Real variance = (onsets - mean).square().sum() /
                    static_cast<Real>(onsets.size() - 1);
    Real std = std::sqrt(variance);
    return onsets / (std + util::tiny<Real>());
}

int round_to_nearest_even(Real value) {
    return static_cast<int>(std::nearbyint(value));
}

// Compute local score with Gaussian weighting
ArrayXr beat_local_score(const ArrayXr& onset_envelope, Real frames_per_beat) {
    Eigen::Index N = onset_envelope.size();
    ArrayXr localscore(N);

    int fpb = round_to_nearest_even(frames_per_beat);
    int window_size = 2 * fpb + 1;

    // Create Gaussian window
    ArrayXr window(window_size);
    for (int k = 0; k < window_size; ++k) {
        Real x = (k - fpb) * 32.0 / fpb;
        window(k) = std::exp(-0.5 * x * x);
    }

    // Same-mode convolution
    for (Eigen::Index i = 0; i < N; ++i) {
        Real sum = 0.0;
        Eigen::Index half_window = window_size / 2;
        Eigen::Index k_start = std::max<Eigen::Index>(
            0, i + half_window - N + 1);
        Eigen::Index k_stop = std::min<Eigen::Index>(
            i + half_window, window_size);

        for (Eigen::Index k = k_start; k < k_stop; ++k) {
            Eigen::Index j = i + window_size / 2 - k;
            sum += window(k) * onset_envelope(j);
        }
        localscore(i) = sum;
    }

    return localscore;
}

// Dynamic programming beat tracker
std::pair<std::vector<int>, ArrayXr> beat_track_dp(
    const ArrayXr& localscore,
    Real frames_per_beat,
    Real tightness) {

    Eigen::Index N = localscore.size();

    std::vector<int> backlink(N, -1);
    ArrayXr cumscore(N);

    Real score_thresh = 0.01 * localscore.maxCoeff();
    bool first_beat = true;

    backlink[0] = -1;
    cumscore(0) = localscore(0);

    int fpb = round_to_nearest_even(frames_per_beat);
    int first_lag = round_to_nearest_even(static_cast<Real>(fpb) / 2.0);

    for (Eigen::Index i = 0; i < N; ++i) {
        Real best_score = -std::numeric_limits<Real>::infinity();
        int beat_location = -1;

        // Search over possible predecessors
        for (Eigen::Index loc = i - first_lag; loc >= i - 2 * fpb; --loc) {
            if (loc < 0) {
                break;
            }
            Real penalty = std::log(static_cast<Real>(i - loc)) - std::log(frames_per_beat);
            Real score = cumscore(loc) - tightness * penalty * penalty;
            if (score > best_score) {
                best_score = score;
                beat_location = static_cast<int>(loc);
            }
        }

        if (beat_location >= 0) {
            cumscore(i) = localscore(i) + best_score;
        } else {
            cumscore(i) = localscore(i);
        }

        if (first_beat && localscore(i) < score_thresh) {
            backlink[i] = -1;
        } else {
            backlink[i] = beat_location;
            first_beat = false;
        }
    }

    return {backlink, cumscore};
}

Real median(std::vector<Real> values) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[mid];
    }

    return 0.5 * (values[mid - 1] + values[mid]);
}

Eigen::Index last_beat(const ArrayXr& cumscore) {
    Eigen::Index N = cumscore.size();
    auto localmax = util::localmax(cumscore);

    std::vector<Real> local_scores;
    local_scores.reserve(static_cast<size_t>(N));
    for (Eigen::Index i = 0; i < N; ++i) {
        if (localmax(i)) {
            local_scores.push_back(cumscore(i));
        }
    }

    Real threshold = 0.5 * median(local_scores);

    Eigen::Index tail = N - 1;
    for (Eigen::Index i = N - 1; i >= 0; --i) {
        if (localmax(i) && cumscore(i) >= threshold) {
            tail = i;
            break;
        }
    }

    return tail;
}

// Backtrack from the best ending point
std::vector<bool> dp_backtrack(const std::vector<int>& backlink, const ArrayXr& cumscore) {
    Eigen::Index N = cumscore.size();
    std::vector<bool> beats(N, false);

    // Backtrack
    Eigen::Index idx = last_beat(cumscore);
    while (idx >= 0) {
        beats[idx] = true;
        idx = backlink[idx];
    }

    return beats;
}

// Trim leading and trailing weak beats
std::vector<bool> trim_beats(const ArrayXr& localscore, const std::vector<bool>& beats, bool trim) {
    std::vector<bool> trimmed = beats;

    if (!trim) {
        return trimmed;
    }

    // Compute the smoothed beat-onset envelope threshold.
    std::vector<Real> beat_scores;
    for (size_t i = 0; i < beats.size(); ++i) {
        if (beats[i]) {
            beat_scores.push_back(localscore(i));
        }
    }

    if (beat_scores.empty()) {
        return trimmed;
    }

    std::vector<Real> window = {0.0, 0.5, 1.0, 0.5, 0.0};
    std::vector<Real> smooth_boe(beat_scores.size() + window.size() - 1, 0.0);
    for (size_t i = 0; i < beat_scores.size(); ++i) {
        for (size_t j = 0; j < window.size(); ++j) {
            smooth_boe[i + j] += beat_scores[i] * window[j];
        }
    }

    size_t start = window.size() / 2;
    size_t stop = std::min(
        smooth_boe.size(),
        static_cast<size_t>(localscore.size()) + window.size() / 2);

    Real mean_square = 0.0;
    size_t smooth_count = 0;
    for (size_t i = start; i < stop; ++i) {
        mean_square += smooth_boe[i] * smooth_boe[i];
        ++smooth_count;
    }

    Real threshold = 0.0;
    if (trim && smooth_count > 0) {
        threshold = 0.5 * std::sqrt(mean_square / static_cast<Real>(smooth_count));
    }

    Eigen::Index n = 0;
    while (n < localscore.size() && localscore(n) <= threshold) {
        trimmed[static_cast<size_t>(n)] = false;
        ++n;
    }

    n = localscore.size() - 1;
    while (n >= 0 && localscore(n) <= threshold) {
        trimmed[static_cast<size_t>(n)] = false;
        --n;
    }

    return trimmed;
}

} // anonymous namespace

std::pair<Real, std::vector<Eigen::Index>> beat_track(
    const ArrayXr& onset_envelope,
    Real sr,
    int hop_length,
    Real start_bpm,
    Real tightness,
    bool trim,
    std::optional<Real> bpm_opt,
    BeatUnits units) {

    // Check for empty signal
    if (onset_envelope.size() == 0 || onset_envelope.maxCoeff() == 0) {
        return {0.0, {}};
    }

    // Estimate BPM if not provided
    Real bpm_val;
    if (bpm_opt.has_value()) {
        bpm_val = bpm_opt.value();
    } else {
        bpm_val = tempo(onset_envelope, sr, hop_length, start_bpm);
    }

    if (bpm_val <= 0) {
        return {0.0, {}};
    }

    // Convert BPM to frames per beat
    Real frame_rate = sr / hop_length;
    Real frames_per_beat = std::nearbyint(frame_rate * 60.0 / bpm_val);

    // Normalize onsets
    ArrayXr normalized = normalize_onsets(onset_envelope);

    // Compute local score
    ArrayXr localscore = beat_local_score(normalized, frames_per_beat);

    // Run DP
    auto [backlink, cumscore] = beat_track_dp(localscore, frames_per_beat, tightness);

    // Backtrack to find beats
    std::vector<bool> beats = dp_backtrack(backlink, cumscore);

    // Trim weak beats
    beats = trim_beats(localscore, beats, trim);

    // Convert to indices
    std::vector<Eigen::Index> beat_frames;
    for (size_t i = 0; i < beats.size(); ++i) {
        if (beats[i]) {
            beat_frames.push_back(i);
        }
    }

    // Convert units
    if (units == BeatUnits::Samples) {
        for (auto& b : beat_frames) {
            b = frames_to_samples(b, hop_length);
        }
    } else if (units == BeatUnits::Time) {
        throw ParameterError(
            "BeatUnits::Time is not supported by beat_track/beat_track_audio; "
            "use beat_track_times for real-valued times.");
    }

    return {bpm_val, beat_frames};
}

std::pair<Real, std::vector<Eigen::Index>> beat_track_audio(
    const ArrayXr& y,
    Real sr,
    int hop_length,
    Real start_bpm,
    Real tightness,
    bool trim,
    std::optional<Real> bpm_opt,
    BeatUnits units) {

    // Compute onset envelope
    ArrayXr envelope = onset::onset_strength(y, sr, 2048, hop_length, 1, 1,
                                             false, true, AggregateFunc::Median);

    return beat_track(envelope, sr, hop_length, start_bpm, tightness, trim, bpm_opt, units);
}

std::pair<Real, ArrayXr> beat_track_times(
    const ArrayXr& y,
    Real sr,
    int hop_length,
    Real start_bpm,
    Real tightness,
    bool trim) {

    auto [bpm, frames] = beat_track_audio(y, sr, hop_length, start_bpm, tightness, trim,
                                           std::nullopt, BeatUnits::Frames);

    ArrayXr times(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        times(i) = frames_to_time(frames[i], sr, hop_length);
    }

    return {bpm, times};
}

// ============================================================================
// Predominant Local Pulse (PLP)
// ============================================================================

ArrayXr plp(const ArrayXr& onset_envelope, Real sr,
            int hop_length, int win_length,
            std::optional<Real> tempo_min,
            std::optional<Real> tempo_max) {

    Eigen::Index n = onset_envelope.size();

    if (n == 0) {
        return ArrayXr(0);
    }

    // Compute Fourier tempogram
    ArrayXXc ftgram = feature::fourier_tempogram(onset_envelope, sr, hop_length,
                                                  win_length, true);

    // Get tempo frequencies for Fourier tempogram bins
    ArrayXr tempo_freqs = fourier_tempo_frequencies(sr, win_length, hop_length);

    // Zero out bins outside [tempo_min, tempo_max]
    for (Eigen::Index i = 0; i < tempo_freqs.size(); ++i) {
        bool out_of_range = false;
        if (tempo_min.has_value() && tempo_freqs(i) < tempo_min.value()) {
            out_of_range = true;
        }
        if (tempo_max.has_value() && tempo_freqs(i) > tempo_max.value()) {
            out_of_range = true;
        }
        if (out_of_range) {
            for (Eigen::Index j = 0; j < ftgram.cols(); ++j) {
                ftgram(i, j) = Complex(0, 0);
            }
        }
    }

    // Find peak bin per frame using log-magnitude
    Eigen::Index n_bins = ftgram.rows();
    Eigen::Index n_frames = ftgram.cols();

    for (Eigen::Index j = 0; j < n_frames; ++j) {
        // Find magnitude peak
        Eigen::Index peak_bin = 0;
        Real peak_val = -std::numeric_limits<Real>::infinity();

        for (Eigen::Index i = 0; i < n_bins; ++i) {
            Real mag = std::abs(ftgram(i, j));
            Real log_mag = std::log1p(1e6 * mag);
            if (log_mag > peak_val) {
                peak_val = log_mag;
                peak_bin = i;
            }
        }

        // Zero out all non-peak bins in this frame
        for (Eigen::Index i = 0; i < n_bins; ++i) {
            if (i != peak_bin) {
                ftgram(i, j) = Complex(0, 0);
            }
        }

        // Normalize the peak bin
        Real peak_mag = std::abs(ftgram(peak_bin, j));
        Real tiny = std::numeric_limits<Real>::min();
        if (peak_mag > tiny) {
            ftgram(peak_bin, j) /= (std::sqrt(tiny) + peak_mag);
        }
    }

    // Inverse STFT with hop_length=1 and n_fft=win_length to get pulse signal
    ArrayXr pulse = istft(ftgram, 1 /* hop_length */, std::nullopt,
                          win_length, WindowType::Hann, true,
                          static_cast<int>(n));

    // Clip to non-negative
    for (Eigen::Index i = 0; i < pulse.size(); ++i) {
        if (pulse(i) < 0) pulse(i) = 0;
    }

    // Normalize to [0, 1]
    Real max_val = pulse.maxCoeff();
    if (max_val > 0) {
        pulse /= max_val;
    }

    return pulse;
}

ArrayXr plp_audio(const ArrayXr& y, Real sr,
                  int hop_length, int win_length,
                  std::optional<Real> tempo_min,
                  std::optional<Real> tempo_max) {

    ArrayXr envelope = onset::onset_strength(y, sr, 2048, hop_length);
    return plp(envelope, sr, hop_length, win_length, tempo_min, tempo_max);
}

} // namespace beat
} // namespace librosa
