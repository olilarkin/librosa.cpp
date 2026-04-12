#include <librosa/effects.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/feature/spectral.hpp>
#include <librosa/decompose.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>
#include <algorithm>

namespace librosa {
namespace effects {

// ============================================================================
// Time Stretching
// ============================================================================

ArrayXr time_stretch(
    const ArrayXr& y,
    Real rate,
    int n_fft,
    std::optional<int> hop_length_opt,
    std::optional<int> win_length,
    WindowType window,
    bool center) {

    if (rate <= 0) {
        throw ParameterError("rate must be a positive number");
    }

    int hop_length = hop_length_opt.value_or(n_fft / 4);

    // Compute STFT
    ArrayXXc D = stft(y, n_fft, hop_length, win_length, window, center);

    // Stretch via phase vocoder
    ArrayXXc D_stretch = phase_vocoder(D, rate, hop_length, n_fft);

    // Predict output length
    Eigen::Index len_stretch = static_cast<Eigen::Index>(std::round(y.size() / rate));

    // Invert STFT
    ArrayXr y_stretch = istft(D_stretch, hop_length, win_length, n_fft, window, center, len_stretch);

    return y_stretch;
}

// ============================================================================
// Pitch Shifting
// ============================================================================

ArrayXr pitch_shift(
    const ArrayXr& y,
    Real sr,
    Real n_steps,
    int bins_per_octave,
    const std::string& res_type,
    int n_fft,
    std::optional<int> hop_length) {

    if (!util::is_positive_int(bins_per_octave)) {
        throw ParameterError("bins_per_octave must be a positive integer");
    }

    // Compute stretch rate
    Real rate = std::pow(2.0, -n_steps / bins_per_octave);

    // Time stretch
    ArrayXr y_shift = time_stretch(y, rate, n_fft, hop_length);

    // Resample to original sample rate
    y_shift = resample(y_shift, sr / rate, sr, res_type);

    // Fix length to match original
    y_shift = util::fix_length(y_shift, y.size());

    return y_shift;
}

// ============================================================================
// Helper: Signal to frame non-silent indicator
// ============================================================================

namespace {

ArrayXr signal_to_frame_nonsilent(
    const ArrayXr& y,
    int frame_length,
    int hop_length,
    Real top_db,
    Real ref) {

    // Compute RMS
    ArrayXXr energy = feature::rms(y, frame_length, hop_length, true);

    // Get reference value
    Real ref_val;
    if (ref < 0) {
        ref_val = energy.maxCoeff();
    } else {
        ref_val = ref;
    }

    // Convert to dB (matching amplitude_to_db with amin applied to both sides)
    Real amin = 1e-10;
    Real ref_db = 10.0 * std::log10(std::max(amin, ref_val * ref_val));
    ArrayXr db(energy.cols());
    for (Eigen::Index i = 0; i < energy.cols(); ++i) {
        Real val = std::max(energy(0, i) * energy(0, i), amin);
        db(i) = 10.0 * std::log10(val) - ref_db;
    }

    // Return non-silent indicator
    ArrayXr non_silent(db.size());
    for (Eigen::Index i = 0; i < db.size(); ++i) {
        non_silent(i) = (db(i) > -top_db) ? 1.0 : 0.0;
    }

    return non_silent;
}

} // anonymous namespace

// ============================================================================
// Trimming
// ============================================================================

std::pair<ArrayXr, std::pair<Eigen::Index, Eigen::Index>> trim(
    const ArrayXr& y,
    Real top_db,
    Real ref,
    int frame_length,
    int hop_length) {

    ArrayXr non_silent = signal_to_frame_nonsilent(y, frame_length, hop_length, top_db, ref);

    // Find first and last non-silent frames
    Eigen::Index first_nonsilent = -1;
    Eigen::Index last_nonsilent = -1;

    for (Eigen::Index i = 0; i < non_silent.size(); ++i) {
        if (non_silent(i) > 0) {
            if (first_nonsilent < 0) {
                first_nonsilent = i;
            }
            last_nonsilent = i;
        }
    }

    Eigen::Index start, end;
    if (first_nonsilent >= 0) {
        start = frames_to_samples(first_nonsilent, hop_length);
        end = std::min(y.size(), frames_to_samples(last_nonsilent + 1, hop_length));
    } else {
        start = 0;
        end = 0;
    }

    // Slice the buffer
    ArrayXr y_trimmed = y.segment(start, end - start);

    return {y_trimmed, {start, end}};
}

// ============================================================================
// Splitting
// ============================================================================

std::vector<std::pair<Eigen::Index, Eigen::Index>> split(
    const ArrayXr& y,
    Real top_db,
    Real ref,
    int frame_length,
    int hop_length) {

    ArrayXr non_silent = signal_to_frame_nonsilent(y, frame_length, hop_length, top_db, ref);

    // Find edges where sign flips
    std::vector<Eigen::Index> edges;

    // Check first frame
    if (non_silent(0) > 0) {
        edges.push_back(0);
    }

    // Find transitions
    for (Eigen::Index i = 1; i < non_silent.size(); ++i) {
        bool prev = non_silent(i - 1) > 0;
        bool curr = non_silent(i) > 0;
        if (prev != curr) {
            edges.push_back(i);
        }
    }

    // Check last frame
    if (non_silent(non_silent.size() - 1) > 0) {
        edges.push_back(non_silent.size());
    }

    // Convert from frames to samples
    std::vector<std::pair<Eigen::Index, Eigen::Index>> intervals;
    for (size_t i = 0; i + 1 < edges.size(); i += 2) {
        Eigen::Index start = frames_to_samples(edges[i], hop_length);
        Eigen::Index end = std::min(y.size(), frames_to_samples(edges[i + 1], hop_length));
        intervals.push_back({start, end});
    }

    return intervals;
}

// ============================================================================
// Preemphasis/Deemphasis
// ============================================================================

ArrayXr preemphasis(const ArrayXr& y, Real coef) {
    if (y.size() == 0) {
        return y;
    }

    ArrayXr result(y.size());
    result(0) = y(0);

    for (Eigen::Index i = 1; i < y.size(); ++i) {
        result(i) = y(i) - coef * y(i - 1);
    }

    return result;
}

ArrayXr deemphasis(const ArrayXr& y, Real coef) {
    if (y.size() == 0) {
        return y;
    }

    ArrayXr result(y.size());
    result(0) = y(0);

    for (Eigen::Index i = 1; i < y.size(); ++i) {
        result(i) = y(i) + coef * result(i - 1);
    }

    return result;
}

// ============================================================================
// Remix
// ============================================================================

ArrayXr remix(
    const ArrayXr& y,
    const std::vector<std::pair<Eigen::Index, Eigen::Index>>& intervals,
    bool align_zeros) {

    if (intervals.empty()) {
        return ArrayXr(0);
    }

    // Find zero crossings if needed
    std::vector<Eigen::Index> zeros;
    if (align_zeros) {
        auto crossings = zero_crossings(y);
        for (Eigen::Index i = 0; i < crossings.size(); ++i) {
            if (crossings(i)) {
                zeros.push_back(i);
            }
        }
        zeros.push_back(y.size());  // Force end-of-signal
    }

    // Calculate total output size
    Eigen::Index total_size = 0;
    std::vector<std::pair<Eigen::Index, Eigen::Index>> adjusted_intervals;

    for (const auto& interval : intervals) {
        Eigen::Index start = interval.first;
        Eigen::Index end = interval.second;

        if (align_zeros && !zeros.empty()) {
            // Find nearest zero crossing for start
            auto it_start = std::lower_bound(zeros.begin(), zeros.end(), start);
            if (it_start != zeros.begin()) {
                auto prev = std::prev(it_start);
                if (it_start == zeros.end() ||
                    std::abs(static_cast<long>(*prev - start)) <= std::abs(static_cast<long>(*it_start - start))) {
                    start = *prev;
                } else {
                    start = *it_start;
                }
            } else if (it_start != zeros.end()) {
                start = *it_start;
            }

            // Find nearest zero crossing for end
            auto it_end = std::lower_bound(zeros.begin(), zeros.end(), end);
            if (it_end != zeros.begin()) {
                auto prev = std::prev(it_end);
                if (it_end == zeros.end() ||
                    std::abs(static_cast<long>(*prev - end)) <= std::abs(static_cast<long>(*it_end - end))) {
                    end = *prev;
                } else {
                    end = *it_end;
                }
            } else if (it_end != zeros.end()) {
                end = *it_end;
            }
        }

        start = std::max(Eigen::Index(0), std::min(start, y.size()));
        end = std::max(start, std::min(end, y.size()));

        adjusted_intervals.push_back({start, end});
        total_size += end - start;
    }

    // Build output
    ArrayXr result(total_size);
    Eigen::Index pos = 0;

    for (const auto& interval : adjusted_intervals) {
        Eigen::Index len = interval.second - interval.first;
        if (len > 0) {
            result.segment(pos, len) = y.segment(interval.first, len);
            pos += len;
        }
    }

    return result;
}

// ============================================================================
// Harmonic/Percussive Separation Effects
// ============================================================================

ArrayXr harmonic(
    const ArrayXr& y,
    int kernel_size,
    Real power,
    bool mask,
    Real margin,
    int n_fft,
    std::optional<int> hop_length_opt,
    std::optional<int> win_length,
    WindowType window,
    bool center,
    PadMode pad_mode) {

    int hop_length = hop_length_opt.value_or(n_fft / 4);

    // Compute STFT
    ArrayXXc D = stft(y, n_fft, hop_length, win_length, window, center, pad_mode);

    // Get magnitude spectrogram
    ArrayXXr S = magnitude(D);

    // Apply HPSS to get harmonic component
    auto [H, P] = decompose::hpss(S, kernel_size, power, mask, margin);

    // Reconstruct phase: multiply harmonic magnitude by original phase
    ArrayXXc D_harmonic(D.rows(), D.cols());
    for (Eigen::Index i = 0; i < D.rows(); ++i) {
        for (Eigen::Index j = 0; j < D.cols(); ++j) {
            Real mag = H(i, j);
            Real orig_mag = std::abs(D(i, j));
            if (orig_mag > 0) {
                D_harmonic(i, j) = D(i, j) * (mag / orig_mag);
            } else {
                D_harmonic(i, j) = Complex(0, 0);
            }
        }
    }

    // Inverse STFT
    return istft(D_harmonic, hop_length, win_length, n_fft, window, center,
                 static_cast<int>(y.size()));
}

ArrayXr percussive(
    const ArrayXr& y,
    int kernel_size,
    Real power,
    bool mask,
    Real margin,
    int n_fft,
    std::optional<int> hop_length_opt,
    std::optional<int> win_length,
    WindowType window,
    bool center,
    PadMode pad_mode) {

    int hop_length = hop_length_opt.value_or(n_fft / 4);

    // Compute STFT
    ArrayXXc D = stft(y, n_fft, hop_length, win_length, window, center, pad_mode);

    // Get magnitude spectrogram
    ArrayXXr S = magnitude(D);

    // Apply HPSS to get percussive component
    auto [H, P] = decompose::hpss(S, kernel_size, power, mask, margin);

    // Reconstruct phase: multiply percussive magnitude by original phase
    ArrayXXc D_percussive(D.rows(), D.cols());
    for (Eigen::Index i = 0; i < D.rows(); ++i) {
        for (Eigen::Index j = 0; j < D.cols(); ++j) {
            Real mag = P(i, j);
            Real orig_mag = std::abs(D(i, j));
            if (orig_mag > 0) {
                D_percussive(i, j) = D(i, j) * (mag / orig_mag);
            } else {
                D_percussive(i, j) = Complex(0, 0);
            }
        }
    }

    // Inverse STFT
    return istft(D_percussive, hop_length, win_length, n_fft, window, center,
                 static_cast<int>(y.size()));
}

} // namespace effects
} // namespace librosa
