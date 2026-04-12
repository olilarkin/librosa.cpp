#include <librosa/filters.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include "internal/rotate2d.hpp"
#include "internal/elliptic.hpp"
#include <cmath>
#include <algorithm>
#include <array>
#include <complex>

namespace librosa {
namespace filters {

// ============================================================================
// Window Bandwidth Constants
// ============================================================================

const std::unordered_map<std::string, Real> WINDOW_BANDWIDTHS = {
    {"bart", 1.3334961334912805},
    {"barthann", 1.4560255965133932},
    {"bartlett", 1.3334961334912805},
    {"bkh", 2.0045975283585014},
    {"black", 1.7269681554262326},
    {"blackharr", 2.0045975283585014},
    {"blackman", 1.7269681554262326},
    {"blackmanharris", 2.0045975283585014},
    {"blk", 1.7269681554262326},
    {"bman", 1.7859588613860062},
    {"bmn", 1.7859588613860062},
    {"bohman", 1.7859588613860062},
    {"box", 1.0},
    {"boxcar", 1.0},
    {"brt", 1.3334961334912805},
    {"brthan", 1.4560255965133932},
    {"bth", 1.4560255965133932},
    {"cosine", 1.2337005350199792},
    {"flat", 2.7762255046484143},
    {"flattop", 2.7762255046484143},
    {"flt", 2.7762255046484143},
    {"halfcosine", 1.2337005350199792},
    {"ham", 1.3629455320350348},
    {"hamm", 1.3629455320350348},
    {"hamming", 1.3629455320350348},
    {"han", 1.50018310546875},
    {"hann", 1.50018310546875},
    {"nut", 1.9763500280946082},
    {"nutl", 1.9763500280946082},
    {"nuttall", 1.9763500280946082},
    {"ones", 1.0},
    {"par", 1.9174603174603191},
    {"parz", 1.9174603174603191},
    {"parzen", 1.9174603174603191},
    {"rect", 1.0},
    {"rectangular", 1.0},
    {"tri", 1.3331706523555851},
    {"triang", 1.3331706523555851},
    {"triangle", 1.3331706523555851},
};

// ============================================================================
// Mel Filter Bank
// ============================================================================

ArrayXXr mel(Real sr, int n_fft, int n_mels,
             Real fmin, std::optional<Real> fmax_opt,
             bool htk, MelNorm norm) {

    Real fmax = fmax_opt.value_or(sr / 2.0);

    // Initialize weights
    ArrayXXr weights = ArrayXXr::Zero(n_mels, 1 + n_fft / 2);

    // Center frequencies of each FFT bin
    ArrayXr fftfreqs = fft_frequencies(sr, n_fft);

    // Center frequencies of mel bands
    ArrayXr mel_f = mel_frequencies(n_mels + 2, fmin, fmax, htk);

    // Compute differences between mel frequencies
    ArrayXr fdiff(n_mels + 1);
    for (Eigen::Index i = 0; i < n_mels + 1; ++i) {
        fdiff(i) = mel_f(i + 1) - mel_f(i);
    }

    // Compute ramps: mel_f[i] - fftfreqs[j] for all i, j
    // ramps has shape (n_mels + 2, n_fft/2 + 1)

    for (int i = 0; i < n_mels; ++i) {
        for (Eigen::Index j = 0; j < fftfreqs.size(); ++j) {
            // Lower slope
            Real lower = -(mel_f(i) - fftfreqs(j)) / fdiff(i);
            // Upper slope
            Real upper = (mel_f(i + 2) - fftfreqs(j)) / fdiff(i + 1);

            // Intersect with each other and zero
            weights(i, j) = std::max(Real(0.0), std::min(lower, upper));
        }
    }

    // Apply normalization
    switch (norm) {
        case MelNorm::Slaney: {
            // Slaney-style: scale to constant energy per channel
            for (int i = 0; i < n_mels; ++i) {
                Real enorm = 2.0 / (mel_f(i + 2) - mel_f(i));
                weights.row(i) *= enorm;
            }
            break;
        }
        case MelNorm::L1: {
            for (int i = 0; i < n_mels; ++i) {
                Real sum = weights.row(i).sum();
                if (sum > 0) {
                    weights.row(i) /= sum;
                }
            }
            break;
        }
        case MelNorm::L2: {
            for (int i = 0; i < n_mels; ++i) {
                Real norm_val = std::sqrt((weights.row(i) * weights.row(i)).sum());
                if (norm_val > 0) {
                    weights.row(i) /= norm_val;
                }
            }
            break;
        }
        case MelNorm::Inf: {
            for (int i = 0; i < n_mels; ++i) {
                Real max_val = weights.row(i).maxCoeff();
                if (max_val > 0) {
                    weights.row(i) /= max_val;
                }
            }
            break;
        }
        case MelNorm::None:
        default:
            // No normalization
            break;
    }

    return weights;
}

// ============================================================================
// Chroma Filter Bank
// ============================================================================

ArrayXXr chroma(Real sr, int n_fft, int n_chroma,
                Real tuning, Real ctroct,
                std::optional<Real> octwidth,
                std::optional<Real> norm_val, bool base_c) {

    ArrayXXr wts = ArrayXXr::Zero(n_chroma, n_fft);

    // Get FFT frequencies (excluding DC)
    ArrayXr frequencies(n_fft - 1);
    for (int i = 1; i < n_fft; ++i) {
        frequencies(i - 1) = static_cast<Real>(i) * sr / n_fft;
    }

    // Map to fractional chroma bins
    ArrayXr frqbins(n_fft - 1);
    for (Eigen::Index i = 0; i < n_fft - 1; ++i) {
        frqbins(i) = n_chroma * hz_to_octs(frequencies(i), tuning, n_chroma);
    }

    // Make up a value for 0 Hz = 1.5 octaves below bin 1
    ArrayXr frqbins_full(n_fft);
    frqbins_full(0) = frqbins(0) - 1.5 * n_chroma;
    frqbins_full.tail(n_fft - 1) = frqbins;

    // Compute bin widths
    ArrayXr binwidthbins(n_fft);
    for (Eigen::Index i = 0; i < n_fft - 1; ++i) {
        binwidthbins(i) = std::max(frqbins_full(i + 1) - frqbins_full(i), Real(1.0));
    }
    binwidthbins(n_fft - 1) = 1.0;

    Real n_chroma2 = std::round(static_cast<Real>(n_chroma) / 2.0);

    // Compute Gaussian bumps
    for (int c = 0; c < n_chroma; ++c) {
        for (int f = 0; f < n_fft; ++f) {
            Real D = frqbins_full(f) - static_cast<Real>(c);

            // Project into range -n_chroma/2 .. n_chroma/2
            D = std::fmod(D + n_chroma2 + 10.0 * n_chroma, static_cast<Real>(n_chroma)) - n_chroma2;

            // Gaussian bump
            Real gauss = std::exp(-0.5 * std::pow(2.0 * D / binwidthbins(f), 2));
            wts(c, f) = gauss;
        }
    }

    // Normalize each column
    if (norm_val.has_value()) {
        Real norm = norm_val.value();
        for (int f = 0; f < n_fft; ++f) {
            Real col_norm;
            if (std::isinf(norm)) {
                col_norm = wts.col(f).maxCoeff();
            } else if (norm == 1.0) {
                col_norm = wts.col(f).sum();
            } else if (norm == 2.0) {
                col_norm = std::sqrt((wts.col(f) * wts.col(f)).sum());
            } else {
                col_norm = std::pow((wts.col(f).abs().pow(norm)).sum(), 1.0 / norm);
            }
            if (col_norm > 0) {
                wts.col(f) /= col_norm;
            }
        }
    }

    // Apply octave width scaling
    if (octwidth.has_value()) {
        Real ow = octwidth.value();
        for (int f = 0; f < n_fft; ++f) {
            Real scale = std::exp(-0.5 * std::pow((frqbins_full(f) / n_chroma - ctroct) / ow, 2));
            wts.col(f) *= scale;
        }
    }

    // Roll to start at C instead of A
    if (base_c) {
        int roll_amount = -3 * (n_chroma / 12);
        // Roll rows
        ArrayXXr wts_rolled = wts;
        for (int i = 0; i < n_chroma; ++i) {
            int new_idx = ((i + roll_amount) % n_chroma + n_chroma) % n_chroma;
            wts_rolled.row(new_idx) = wts.row(i);
        }
        wts = wts_rolled;
    }

    // Return only positive frequencies
    return wts.leftCols(1 + n_fft / 2);
}

// ============================================================================
// Window Bandwidth
// ============================================================================

Real window_bandwidth(WindowType window, int n) {
    // Map window type to string key
    std::string key;
    switch (window) {
        case WindowType::Hann:
            key = "hann";
            break;
        case WindowType::Hamming:
            key = "hamming";
            break;
        case WindowType::Blackman:
            key = "blackman";
            break;
        case WindowType::Bartlett:
            key = "bartlett";
            break;
        case WindowType::Rectangular:
            key = "rectangular";
            break;
        default:
            key = "hann";
    }

    return window_bandwidth(key, n);
}

Real window_bandwidth(const std::string& window, int n) {
    // Check if we have a pre-computed value
    auto it = WINDOW_BANDWIDTHS.find(window);
    if (it != WINDOW_BANDWIDTHS.end()) {
        return it->second;
    }

    // Compute bandwidth from window
    ArrayXr win = get_window(window, n, true);
    Real sum_sq = (win * win).sum();
    Real sum = win.sum();

    return n * sum_sq / (sum * sum + util::tiny(sum));
}

// ============================================================================
// CQ to Chroma Transformation
// ============================================================================

ArrayXXr cq_to_chroma(int n_input, int bins_per_octave,
                       int n_chroma, std::optional<Real> fmin_opt,
                       bool base_c) {

    Real fmin = fmin_opt.value_or(note_to_hz("C1"));

    // Check compatibility
    Real n_merge = static_cast<Real>(bins_per_octave) / n_chroma;
    if (std::fmod(n_merge, 1.0) != 0) {
        throw ParameterError("Incompatible CQ merge: input bins must be integer multiple of output bins");
    }

    int n_merge_int = static_cast<int>(n_merge);

    // Create identity repeated n_merge times
    ArrayXXr cq_to_ch = ArrayXXr::Zero(n_chroma, n_chroma * n_merge_int);
    for (int i = 0; i < n_chroma; ++i) {
        for (int j = 0; j < n_merge_int; ++j) {
            cq_to_ch(i, i * n_merge_int + j) = 1.0;
        }
    }

    // Roll left to center
    int roll_cols = n_merge_int / 2;
    if (roll_cols > 0) {
        ArrayXXr temp = cq_to_ch;
        for (int j = 0; j < cq_to_ch.cols(); ++j) {
            int new_j = (j - roll_cols + cq_to_ch.cols()) % cq_to_ch.cols();
            cq_to_ch.col(new_j) = temp.col(j);
        }
    }

    // Repeat for multiple octaves
    int n_octaves = static_cast<int>(std::ceil(static_cast<Real>(n_input) / bins_per_octave));
    ArrayXXr result = ArrayXXr::Zero(n_chroma, n_input);

    for (int oct = 0; oct < n_octaves; ++oct) {
        Eigen::Index start_col = oct * bins_per_octave;
        Eigen::Index cols_to_copy = std::min(static_cast<Eigen::Index>(bins_per_octave),
                                              static_cast<Eigen::Index>(n_input) - start_col);
        if (cols_to_copy > 0) {
            result.block(0, start_col, n_chroma, cols_to_copy) =
                cq_to_ch.block(0, 0, n_chroma, cols_to_copy);
        }
    }

    // Compute roll amount based on fmin
    Real midi_0 = std::fmod(hz_to_midi(fmin), 12.0);
    Real roll;
    if (base_c) {
        roll = midi_0;
    } else {
        roll = midi_0 - 9;
    }
    roll = std::round(roll * (static_cast<Real>(n_chroma) / 12.0));

    // Apply row roll
    int roll_rows = static_cast<int>(roll);
    if (roll_rows != 0) {
        ArrayXXr temp = result;
        for (int i = 0; i < n_chroma; ++i) {
            int new_i = (i + roll_rows + n_chroma) % n_chroma;
            result.row(new_i) = temp.row(i);
        }
    }

    return result;
}

// ============================================================================
// Wavelet Lengths
// ============================================================================

ArrayXr relative_bandwidth(const ArrayXr& freqs) {
    if (freqs.size() <= 1) {
        throw ParameterError("2 or more frequencies required to compute bandwidths");
    }

    ArrayXr bpo = ArrayXr::Zero(freqs.size());
    ArrayXr logf = freqs.log() / std::log(2.0);

    // Reflect at boundaries
    bpo(0) = 1.0 / (logf(1) - logf(0));
    bpo(bpo.size() - 1) = 1.0 / (logf(logf.size() - 1) - logf(logf.size() - 2));

    // Centered difference for interior
    for (Eigen::Index i = 1; i < bpo.size() - 1; ++i) {
        bpo(i) = 2.0 / (logf(i + 1) - logf(i - 1));
    }

    // Compute relative bandwidths
    ArrayXr alpha = (Eigen::pow(2.0, 2.0 / bpo) - 1.0) /
                    (Eigen::pow(2.0, 2.0 / bpo) + 1.0);
    return alpha;
}

std::pair<ArrayXr, Real> wavelet_lengths(
    const ArrayXr& freqs, Real sr, WindowType window,
    Real filter_scale, Real gamma, std::optional<Real> alpha_opt) {

    if (filter_scale <= 0) {
        throw ParameterError("filter_scale must be positive");
    }

    if (gamma < 0) {
        throw ParameterError("gamma must be non-negative");
    }

    if ((freqs <= 0).any()) {
        throw ParameterError("frequencies must be strictly positive");
    }

    // Check ascending order
    for (Eigen::Index i = 0; i < freqs.size() - 1; ++i) {
        if (freqs(i) >= freqs(i + 1)) {
            throw ParameterError("frequencies must be in ascending order");
        }
    }

    ArrayXr alpha;
    if (alpha_opt.has_value()) {
        alpha = ArrayXr::Constant(freqs.size(), alpha_opt.value());
    } else {
        alpha = relative_bandwidth(freqs);
    }

    // Q factor
    ArrayXr Q = filter_scale / alpha;

    // Window bandwidth
    Real wb = window_bandwidth(window);

    // Cutoff frequency
    Real f_cutoff = (freqs * (1.0 + 0.5 * wb / Q) + 0.5 * gamma).maxCoeff();

    // Filter lengths
    ArrayXr lengths = Q * sr / (freqs + gamma / alpha);

    return {lengths, f_cutoff};
}

// ============================================================================
// Wavelet Basis
// ============================================================================

std::pair<ArrayXXc, ArrayXr> wavelet(
    const ArrayXr& freqs, Real sr, WindowType window,
    Real filter_scale, bool pad_fft, std::optional<Real> norm,
    Real gamma, std::optional<Real> alpha_opt) {

    auto [lengths, f_cutoff] = wavelet_lengths(freqs, sr, window, filter_scale, gamma, alpha_opt);

    Eigen::Index n_bins = freqs.size();

    // Build filters
    std::vector<ArrayXc> filters_list;
    Real max_len = 0;

    for (Eigen::Index i = 0; i < n_bins; ++i) {
        Real ilen = lengths(i);

        // Match Python: np.arange(-ilen // 2, ilen // 2, dtype=float)
        // Python // is floor division: -ilen // 2 = floor(-ilen/2) = -ceil(ilen/2)
        //                               ilen // 2 = floor(ilen/2)
        int start = -static_cast<int>(std::ceil(ilen / 2.0));
        int stop = static_cast<int>(std::floor(ilen / 2.0));
        int len = stop - start;  // number of integer-spaced samples

        // Create phasor with integer time indices
        ArrayXr t(len);
        for (int j = 0; j < len; ++j) {
            t(j) = static_cast<Real>(start + j);
        }
        ArrayXr angles = t * 2.0 * constants::PI * freqs(i) / sr;
        ArrayXc sig = util::phasor(angles, 1.0);

        // Apply window: __float_window(window)(len) where len is integer,
        // so it just creates a standard periodic window of size len
        ArrayXr win = get_window(window, len, true);

        for (int j = 0; j < len; ++j) {
            sig(j) *= win(j);
        }

        // Normalize
        if (norm.has_value()) {
            Real n = norm.value();
            Real sig_norm;
            if (std::isinf(n)) {
                sig_norm = sig.abs().maxCoeff();
            } else if (n == 1.0) {
                sig_norm = sig.abs().sum();
            } else if (n == 2.0) {
                sig_norm = std::sqrt((sig.abs() * sig.abs()).sum());
            } else {
                sig_norm = std::pow((sig.abs().pow(n)).sum(), 1.0 / n);
            }
            if (sig_norm > 0) {
                sig /= sig_norm;
            }
        }

        filters_list.push_back(sig);
        max_len = std::max(max_len, static_cast<Real>(len));
    }

    // Determine final length
    int final_len;
    if (pad_fft) {
        final_len = static_cast<int>(std::pow(2, std::ceil(std::log2(max_len))));
    } else {
        final_len = static_cast<int>(std::ceil(max_len));
    }

    // Stack and pad filters
    ArrayXXc filters = ArrayXXc::Zero(n_bins, final_len);
    for (Eigen::Index i = 0; i < n_bins; ++i) {
        const ArrayXc& filt = filters_list[i];
        Eigen::Index pad_left = (final_len - filt.size()) / 2;
        filters.row(i).segment(pad_left, filt.size()) = filt.transpose();
    }

    return {filters, lengths};
}

// ============================================================================
// Wavelet Lengths (array alpha, optional gamma overload)
// ============================================================================

std::pair<ArrayXr, Real> wavelet_lengths(
    const ArrayXr& freqs, Real sr, WindowType window,
    Real filter_scale, std::optional<Real> gamma_opt,
    const ArrayXr& alpha) {

    if (filter_scale <= 0) {
        throw ParameterError("filter_scale must be positive");
    }

    if (gamma_opt.has_value() && gamma_opt.value() < 0) {
        throw ParameterError("gamma must be non-negative");
    }

    if ((freqs <= 0).any()) {
        throw ParameterError("frequencies must be strictly positive");
    }

    // Check ascending order
    for (Eigen::Index i = 0; i < freqs.size() - 1; ++i) {
        if (freqs(i) >= freqs(i + 1)) {
            throw ParameterError("frequencies must be in ascending order");
        }
    }

    // Compute gamma: if nullopt, derive from ERB
    ArrayXr gamma_arr;
    if (gamma_opt.has_value()) {
        gamma_arr = ArrayXr::Constant(freqs.size(), gamma_opt.value());
    } else {
        // gamma = alpha * 24.7 / 0.108
        gamma_arr = alpha * 24.7 / 0.108;
    }

    // Q factor
    ArrayXr Q = filter_scale / alpha;

    // Window bandwidth
    Real wb = window_bandwidth(window);

    // Cutoff frequency
    Real f_cutoff = (freqs * (1.0 + 0.5 * wb / Q) + 0.5 * gamma_arr).maxCoeff();

    // Filter lengths
    ArrayXr lengths = Q * sr / (freqs + gamma_arr / alpha);

    return {lengths, f_cutoff};
}

// ============================================================================
// Wavelet Basis (array alpha, optional gamma overload)
// ============================================================================

std::pair<ArrayXXc, ArrayXr> wavelet(
    const ArrayXr& freqs, Real sr, WindowType window,
    Real filter_scale, bool pad_fft, std::optional<Real> norm,
    std::optional<Real> gamma_opt, const ArrayXr& alpha) {

    auto [lengths, f_cutoff] = wavelet_lengths(freqs, sr, window, filter_scale, gamma_opt, alpha);

    Eigen::Index n_bins = freqs.size();

    // Build filters
    std::vector<ArrayXc> filters_list;
    Real max_len = 0;

    for (Eigen::Index i = 0; i < n_bins; ++i) {
        Real ilen = lengths(i);

        // Match Python: np.arange(-ilen // 2, ilen // 2, dtype=float)
        // Python // is floor division: -ilen // 2 = floor(-ilen/2) = -ceil(ilen/2)
        //                               ilen // 2 = floor(ilen/2)
        int start = -static_cast<int>(std::ceil(ilen / 2.0));
        int stop = static_cast<int>(std::floor(ilen / 2.0));
        int len = stop - start;  // number of integer-spaced samples

        // Create phasor with integer time indices
        ArrayXr t(len);
        for (int j = 0; j < len; ++j) {
            t(j) = static_cast<Real>(start + j);
        }
        ArrayXr angles = t * 2.0 * constants::PI * freqs(i) / sr;
        ArrayXc sig = util::phasor(angles, 1.0);

        // Apply window
        ArrayXr win = get_window(window, len, true);

        for (int j = 0; j < len; ++j) {
            sig(j) *= win(j);
        }

        // Normalize
        if (norm.has_value()) {
            Real n = norm.value();
            Real sig_norm;
            if (std::isinf(n)) {
                sig_norm = sig.abs().maxCoeff();
            } else if (n == 1.0) {
                sig_norm = sig.abs().sum();
            } else if (n == 2.0) {
                sig_norm = std::sqrt((sig.abs() * sig.abs()).sum());
            } else {
                sig_norm = std::pow((sig.abs().pow(n)).sum(), 1.0 / n);
            }
            if (sig_norm > 0) {
                sig /= sig_norm;
            }
        }

        filters_list.push_back(sig);
        max_len = std::max(max_len, static_cast<Real>(len));
    }

    // Determine final length
    int final_len;
    if (pad_fft) {
        final_len = static_cast<int>(std::pow(2, std::ceil(std::log2(max_len))));
    } else {
        final_len = static_cast<int>(std::ceil(max_len));
    }

    // Stack and pad filters
    ArrayXXc filters = ArrayXXc::Zero(n_bins, final_len);
    for (Eigen::Index i = 0; i < n_bins; ++i) {
        const ArrayXc& filt = filters_list[i];
        Eigen::Index pad_left = (final_len - filt.size()) / 2;
        filters.row(i).segment(pad_left, filt.size()) = filt.transpose();
    }

    return {filters, lengths};
}

// ============================================================================
// Diagonal Filter
// ============================================================================

ArrayXXr diagonal_filter(WindowType window, int n,
                          Real slope, std::optional<Real> angle_opt,
                          bool zero_mean) {

    Real angle = angle_opt.value_or(std::atan(slope));

    // Create diagonal window (at 45 degrees = pi/4)
    ArrayXr win_1d = get_window(window, n, false);
    ArrayXXr win = ArrayXXr::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        win(i, i) = win_1d(i);
    }

    // If angle is not 45 degrees, rotate to the desired angle
    Real target_angle = constants::PI / 4.0;  // The diagonal is already at 45 degrees
    if (std::abs(angle - target_angle) > 1e-10) {
        // Rotate by the difference: angle - pi/4
        // scipy.ndimage.rotate uses degrees and rotates CCW
        // Our rotate2d takes radians; a positive value rotates CCW
        Real rotation = -(angle - target_angle);
        win = internal::rotate2d(win, rotation);
    }

    // Clip to non-negative
    win = win.max(0);

    // Normalize
    Real win_sum = win.sum();
    if (win_sum > 0) {
        win /= win_sum;
    }

    if (zero_mean) {
        win -= win.mean();
    }

    return win;
}

// ============================================================================
// Multi-rate Frequencies
// ============================================================================

std::pair<ArrayXr, ArrayXr> mr_frequencies(Real tuning) {
    // MIDI notes from 24+tuning to 108+tuning (C0 to B7)
    ArrayXr midi_notes = ArrayXr::LinSpaced(85, 24 + tuning, 108 + tuning);
    ArrayXr center_freqs = midi_to_hz(midi_notes);

    // Sample rates for different frequency ranges
    ArrayXr sample_rates(85);

    // Notes 0-35: 882 Hz
    for (int i = 0; i < 36; ++i) {
        sample_rates(i) = 882.0;
    }
    // Notes 36-69: 4410 Hz
    for (int i = 36; i < 70; ++i) {
        sample_rates(i) = 4410.0;
    }
    // Notes 70-84: 22050 Hz
    for (int i = 70; i < 85; ++i) {
        sample_rates(i) = 22050.0;
    }

    return {center_freqs, sample_rates};
}

// ============================================================================
// IIR Elliptic Bandpass Filter Design
// ============================================================================

namespace {

using internal::elliptic::ellipk;
using internal::elliptic::ellipj;
using internal::elliptic::arc_jac_sc1;
using internal::elliptic::ellipdeg;

/// ZPK representation for filter design
struct ZPK {
    std::vector<Complex> z; // zeros
    std::vector<Complex> p; // poles
    Real k;                 // gain
};

/// Elliptic analog lowpass prototype (zeros, poles, gain).
/// Port of scipy.signal.ellipap.
ZPK ellipap(int N, Real rp, Real rs) {
    constexpr Real EPSILON = 2e-16;

    if (N == 0) {
        return {{}, {}, std::pow(10.0, -rp / 20.0)};
    }

    Real eps_sq = std::expm1(std::log(10.0) * 0.1 * rp);  // 10^(0.1*rp) - 1
    Real eps = std::sqrt(eps_sq);

    Real rs_pow = std::expm1(std::log(10.0) * 0.1 * rs);  // 10^(0.1*rs) - 1
    Real ck1_sq = eps_sq / rs_pow;

    if (N == 1) {
        Real p_val = -std::sqrt(1.0 / eps_sq);
        return {{}, {Complex(p_val, 0.0)}, -p_val};
    }

    Real m = ellipdeg(N, ck1_sq);
    Real capk = ellipk(m);

    // Compute zeros: j = 1, 3, 5, ..., < N  (i.e., 1-N%2, 1-N%2+2, ...)
    std::vector<int> j_indices;
    for (int j = 1 - (N % 2); j < N; j += 2) {
        j_indices.push_back(j);
    }

    std::vector<Complex> zeros;
    for (int j : j_indices) {
        auto [sn, cn, dn] = ellipj(static_cast<Real>(j) * capk / N, m);
        if (std::abs(sn) > EPSILON) {
            Complex z_val = Complex(0.0, 1.0 / (std::sqrt(m) * sn));
            zeros.push_back(z_val);
            zeros.push_back(std::conj(z_val));
        }
    }

    // Compute poles using v0
    Real K1 = ellipk(ck1_sq);
    Real K1p = ellipk(1.0 - ck1_sq);
    Real r = arc_jac_sc1(1.0 / eps, ck1_sq);
    Real v0 = capk * r / (N * K1);

    auto [sv, cv, dv, ignore] = [&]() {
        auto res = ellipj(v0, 1.0 - m);
        struct { Real sv, cv, dv, phi; } r = {res.sn, res.cn, res.dn, 0.0};
        return r;
    }();

    std::vector<Complex> poles;
    for (int j : j_indices) {
        auto [s, c, d] = ellipj(static_cast<Real>(j) * capk / N, m);
        Real denom = 1.0 - (d * sv) * (d * sv);
        Complex p_val = Complex(-(c * d * sv * cv), s * dv) / denom;
        poles.push_back(p_val);
    }

    // Separate purely real poles (for odd N) and conjugate-pair the rest
    std::vector<Complex> all_poles;
    for (const auto& p : poles) {
        if (std::abs(p.imag()) > EPSILON * std::sqrt(std::norm(p))) {
            // This pole has significant imaginary part → add it and its conjugate
            all_poles.push_back(p);
            all_poles.push_back(std::conj(p));
        } else {
            // Purely real pole (from odd N) → add once
            all_poles.push_back(Complex(p.real(), 0.0));
        }
    }

    // Compute gain
    Complex k_num(1.0, 0.0);
    for (const auto& p : all_poles) k_num *= -p;
    Complex k_den(1.0, 0.0);
    for (const auto& z : zeros) k_den *= -z;
    Real gain = (k_num / k_den).real();

    if (N % 2 == 0) {
        gain /= std::sqrt(1.0 + eps_sq);
    }

    return {zeros, all_poles, gain};
}

/// Compute minimum elliptic filter order for bandpass specification.
/// Port of the ellipord bandpass path from scipy.signal.ellipord.
int ellipord_bandpass(Real wp_low, Real wp_high, Real ws_low, Real ws_high,
                      Real gpass, Real gstop) {
    // Pre-warp to analog domain: warped = 2*fs*tan(pi*W/fs) with fs=2
    Real Wp_low = 4.0 * std::tan(constants::PI * wp_low / 2.0);
    Real Wp_high = 4.0 * std::tan(constants::PI * wp_high / 2.0);
    Real Ws_low = 4.0 * std::tan(constants::PI * ws_low / 2.0);
    Real Ws_high = 4.0 * std::tan(constants::PI * ws_high / 2.0);

    // Transform to lowpass prototype frequency
    Real W0 = std::sqrt(Wp_low * Wp_high);
    Real Bw = Wp_high - Wp_low;

    // Map stopband edges to lowpass prototype
    Real nat1 = std::abs((Ws_low * Ws_low - W0 * W0) / (Ws_low * Bw));
    Real nat2 = std::abs((Ws_high * Ws_high - W0 * W0) / (Ws_high * Bw));
    Real nat = std::min(nat1, nat2);

    // Compute order using elliptic integrals
    Real arg0 = 1.0 / nat;
    Real arg0_sq = arg0 * arg0;
    Real arg1_sq = std::expm1(std::log(10.0) * 0.1 * gpass)
                 / std::expm1(std::log(10.0) * 0.1 * gstop);

    Real d0_K = ellipk(arg0_sq);
    Real d0_Kp = ellipk(1.0 - arg0_sq);
    Real d1_K = ellipk(arg1_sq);
    Real d1_Kp = ellipk(1.0 - arg1_sq);

    int N = static_cast<int>(std::ceil(d0_K * d1_Kp / (d0_Kp * d1_K)));
    if (N < 1) N = 1;
    return N;
}

/// Lowpass to bandpass transformation in ZPK domain (analog).
/// Handles both finite zeros (elliptic) and zeros at infinity (Butterworth).
/// Port of scipy.signal.lp2bp_zpk.
ZPK lp2bp_zpk(const ZPK& lp, Real w0, Real bw) {
    int degree = static_cast<int>(lp.p.size()) - static_cast<int>(lp.z.size());

    // Transform finite zeros
    std::vector<Complex> z_bp;
    for (const auto& zl : lp.z) {
        Complex a = Complex(bw / 2.0, 0.0) * zl;
        Complex sq = std::sqrt(a * a - Complex(w0 * w0, 0.0));
        z_bp.push_back(a + sq);
        z_bp.push_back(a - sq);
    }

    // Transform poles
    std::vector<Complex> p_bp;
    for (const auto& pl : lp.p) {
        Complex a = Complex(bw / 2.0, 0.0) * pl;
        Complex sq = std::sqrt(a * a - Complex(w0 * w0, 0.0));
        p_bp.push_back(a + sq);
        p_bp.push_back(a - sq);
    }

    // Add 'degree' zeros at origin (for the zeros-at-infinity in the prototype)
    for (int i = 0; i < degree; ++i) {
        z_bp.push_back(Complex(0.0, 0.0));
    }

    Real k_bp = lp.k * std::pow(bw, degree);

    return {z_bp, p_bp, k_bp};
}

/// Bilinear transform: analog ZPK → digital ZPK
ZPK bilinear_zpk(const ZPK& analog, Real fs) {
    Real fs2 = 2.0 * fs;

    int degree = static_cast<int>(analog.p.size());
    int num_z = static_cast<int>(analog.z.size());

    // Transform zeros: z_d = (1 + z_a/fs2) / (1 - z_a/fs2)
    std::vector<Complex> z_d;
    z_d.reserve(degree);
    for (const auto& za : analog.z) {
        z_d.push_back((Complex(1.0, 0.0) + za / fs2) / (Complex(1.0, 0.0) - za / fs2));
    }
    // Any zeros at infinity become zeros at z = -1
    for (int i = num_z; i < degree; ++i) {
        z_d.push_back(Complex(-1.0, 0.0));
    }

    // Transform poles: p_d = (1 + p_a/fs2) / (1 - p_a/fs2)
    std::vector<Complex> p_d;
    p_d.reserve(degree);
    for (const auto& pa : analog.p) {
        p_d.push_back((Complex(1.0, 0.0) + pa / fs2) / (Complex(1.0, 0.0) - pa / fs2));
    }

    // Gain adjustment: k_d = k * real(prod(fs2 - z_a) / prod(fs2 - p_a))
    // Only finite analog zeros contribute — zeros at infinity are handled
    // by appending z=-1 above but do NOT affect the gain computation.
    Complex k_num(analog.k, 0.0);
    for (const auto& za : analog.z) {
        k_num *= (fs2 - za);
    }
    Complex k_den(1.0, 0.0);
    for (const auto& pa : analog.p) {
        k_den *= (fs2 - pa);
    }
    Real k_d = (k_num / k_den).real();

    return {z_d, p_d, k_d};
}

/// Convert ZPK to SOS (second-order sections).
/// Groups conjugate pairs for both zeros and poles, sorts poles by distance
/// from unit circle (farthest first for stability), pairs nearest zero/pole
/// groups, and applies gain to first section only.
SOSFilter zpk2sos(const ZPK& zpk) {
    int n_sections = (static_cast<int>(zpk.p.size()) + 1) / 2;
    if (n_sections == 0) {
        SOSSection sec = {zpk.k, 0.0, 0.0, 1.0, 0.0, 0.0};
        return {sec};
    }

    // Helper: group complex values into conjugate pairs
    // Returns pairs. For real values, pairs with next real value.
    struct CPair {
        Complex v1, v2;
    };

    auto group_conjugates = [](const std::vector<Complex>& vals) -> std::vector<CPair> {
        std::vector<CPair> pairs;
        std::vector<bool> used(vals.size(), false);

        // First: pair conjugates
        for (size_t i = 0; i < vals.size(); ++i) {
            if (used[i]) continue;
            if (std::abs(vals[i].imag()) < 1e-14) continue; // skip reals
            for (size_t j = i + 1; j < vals.size(); ++j) {
                if (used[j]) continue;
                if (std::abs(vals[i] - std::conj(vals[j])) < 1e-10) {
                    used[i] = used[j] = true;
                    pairs.push_back({vals[i], vals[j]});
                    break;
                }
            }
        }

        // Then: pair remaining reals
        std::vector<size_t> reals;
        for (size_t i = 0; i < vals.size(); ++i) {
            if (!used[i]) reals.push_back(i);
        }
        for (size_t i = 0; i + 1 < reals.size(); i += 2) {
            pairs.push_back({vals[reals[i]], vals[reals[i + 1]]});
        }
        // If odd one left, it will be a single real (shouldn't happen for well-formed filters)
        if (reals.size() % 2 == 1) {
            pairs.push_back({vals[reals.back()], Complex(0.0, 0.0)});
        }

        return pairs;
    };

    // Pad zeros to match poles count
    std::vector<Complex> all_zeros = zpk.z;
    for (int i = static_cast<int>(zpk.z.size()); i < static_cast<int>(zpk.p.size()); ++i) {
        all_zeros.push_back(Complex(0.0, 0.0));
    }

    auto zero_pairs = group_conjugates(all_zeros);
    auto pole_pairs = group_conjugates(zpk.p);

    // Sort pole pairs: farthest from unit circle first (most stable)
    std::sort(pole_pairs.begin(), pole_pairs.end(),
              [](const CPair& a, const CPair& b) {
                  Real da = std::abs(std::abs(a.v1) - 1.0);
                  Real db = std::abs(std::abs(b.v1) - 1.0);
                  return da > db;
              });

    // For each pole pair, find the nearest zero pair (by angular proximity)
    // This matches scipy's pairing heuristic
    std::vector<bool> zero_used(zero_pairs.size(), false);

    SOSFilter sos;
    for (const auto& pp : pole_pairs) {
        SOSSection sec;

        // Denominator
        Complex psum = pp.v1 + pp.v2;
        Complex pprod = pp.v1 * pp.v2;
        sec[3] = 1.0;
        sec[4] = -psum.real();
        sec[5] = pprod.real();

        // Find nearest unused zero pair
        int best_z = -1;
        Real best_dist = std::numeric_limits<Real>::infinity();
        Real pole_angle = std::arg(pp.v1);

        for (size_t zi = 0; zi < zero_pairs.size(); ++zi) {
            if (zero_used[zi]) continue;
            Real zero_angle = std::arg(zero_pairs[zi].v1);
            Real dist = std::abs(pole_angle - zero_angle);
            if (dist > constants::PI) dist = 2.0 * constants::PI - dist;
            if (dist < best_dist) {
                best_dist = dist;
                best_z = static_cast<int>(zi);
            }
        }

        if (best_z >= 0) {
            zero_used[best_z] = true;
            Complex z1 = zero_pairs[best_z].v1;
            Complex z2 = zero_pairs[best_z].v2;
            Complex zsum = z1 + z2;
            Complex zprod = z1 * z2;
            sec[0] = 1.0;
            sec[1] = -zsum.real();
            sec[2] = zprod.real();
        } else {
            sec[0] = 1.0;
            sec[1] = 0.0;
            sec[2] = 0.0;
        }

        sos.push_back(sec);
    }

    // Apply gain to first section
    if (!sos.empty()) {
        sos[0][0] *= zpk.k;
        sos[0][1] *= zpk.k;
        sos[0][2] *= zpk.k;
    }

    return sos;
}

/// Design an elliptic bandpass IIR filter in SOS form
/// wp_low, wp_high: passband edges (normalized to Nyquist, 0-1)
/// ws_low, ws_high: stopband edges (normalized to Nyquist, 0-1)
SOSFilter iir_bandpass_ellip(Real wp_low, Real wp_high,
                              Real ws_low, Real ws_high,
                              Real gpass, Real gstop) {
    // Clamp to valid range
    wp_low = std::max(wp_low, 1e-6);
    wp_high = std::min(wp_high, 1.0 - 1e-6);
    ws_low = std::max(ws_low, 1e-6);
    ws_high = std::min(ws_high, 1.0 - 1e-6);

    // 1. Compute minimum elliptic filter order
    int N = ellipord_bandpass(wp_low, wp_high, ws_low, ws_high, gpass, gstop);

    // 2. Elliptic analog lowpass prototype
    ZPK proto = ellipap(N, gpass, gstop);

    // 3. Pre-warp passband edges to analog domain: warped = 2*fs*tan(pi*W/fs) with fs=2
    Real Wp_low = 4.0 * std::tan(constants::PI * wp_low / 2.0);
    Real Wp_high = 4.0 * std::tan(constants::PI * wp_high / 2.0);
    Real w0 = std::sqrt(Wp_low * Wp_high);
    Real bw = Wp_high - Wp_low;

    // 4. LP→BP transform (handles finite zeros from elliptic prototype)
    ZPK analog_bp = lp2bp_zpk(proto, w0, bw);

    // 5. Bilinear transform (fs=2 for normalized frequencies)
    ZPK digital = bilinear_zpk(analog_bp, 2.0);

    // 6. Convert to SOS
    return zpk2sos(digital);
}

} // anonymous namespace

// ============================================================================
// Semitone Filterbank
// ============================================================================

std::pair<std::vector<SOSFilter>, ArrayXr> semitone_filterbank(
    std::optional<ArrayXr> center_freqs_opt,
    Real tuning,
    std::optional<ArrayXr> sample_rates_opt,
    Real Q,
    Real passband_ripple,
    Real stopband_attenuation) {

    ArrayXr center_freqs, sample_rates;

    if (!center_freqs_opt.has_value() && !sample_rates_opt.has_value()) {
        auto [cf, sr] = mr_frequencies(tuning);
        center_freqs = cf;
        sample_rates = sr;
    } else {
        if (!center_freqs_opt.has_value()) {
            throw ParameterError("center_freqs must be provided");
        }
        if (!sample_rates_opt.has_value()) {
            throw ParameterError("sample_rates must be provided");
        }
        center_freqs = center_freqs_opt.value();
        sample_rates = sample_rates_opt.value();
    }

    if (center_freqs.size() != sample_rates.size()) {
        throw ParameterError("Number of center_freqs and sample_rates must be equal");
    }

    Eigen::Index n = center_freqs.size();
    std::vector<SOSFilter> filterbank;
    filterbank.reserve(n);

    for (Eigen::Index i = 0; i < n; ++i) {
        Real cur_center = center_freqs(i);
        Real cur_sr = sample_rates(i);
        Real cur_nyquist = 0.5 * cur_sr;
        Real cur_bw = cur_center / Q;

        // Passband edges normalized by Nyquist
        Real wp_low = (cur_center - 0.5 * cur_bw) / cur_nyquist;
        Real wp_high = (cur_center + 0.5 * cur_bw) / cur_nyquist;

        // Stopband edges normalized by Nyquist
        Real ws_low = (cur_center - cur_bw) / cur_nyquist;
        Real ws_high = (cur_center + cur_bw) / cur_nyquist;

        SOSFilter filter = iir_bandpass_ellip(wp_low, wp_high,
                                                ws_low, ws_high,
                                                passband_ripple,
                                                stopband_attenuation);
        filterbank.push_back(std::move(filter));
    }

    return {filterbank, sample_rates};
}

// ============================================================================
// SOS Filtering
// ============================================================================

ArrayXr sosfilt(const SOSFilter& sos, const ArrayXr& x) {
    ArrayXr y = x;
    Eigen::Index n = x.size();

    for (const auto& sec : sos) {
        Real b0 = sec[0], b1 = sec[1], b2 = sec[2];
        Real a1 = sec[4], a2 = sec[5];
        // a0 = sec[3] assumed to be 1.0

        Real z1 = 0.0, z2 = 0.0;
        for (Eigen::Index i = 0; i < n; ++i) {
            Real xi = y(i);
            Real yi = b0 * xi + z1;
            z1 = b1 * xi - a1 * yi + z2;
            z2 = b2 * xi - a2 * yi;
            y(i) = yi;
        }
    }

    return y;
}

std::pair<ArrayXr, ArrayXXr> sosfilt(const SOSFilter& sos, const ArrayXr& x,
                                     const ArrayXXr& zi) {
    int n_sections = static_cast<int>(sos.size());
    Eigen::Index n = x.size();
    ArrayXr y = x;
    ArrayXXr zf(n_sections, 2);

    for (int s = 0; s < n_sections; ++s) {
        Real b0 = sos[s][0], b1 = sos[s][1], b2 = sos[s][2];
        Real a1 = sos[s][4], a2 = sos[s][5];

        Real z1 = zi(s, 0), z2 = zi(s, 1);
        for (Eigen::Index i = 0; i < n; ++i) {
            Real xi = y(i);
            Real yi = b0 * xi + z1;
            z1 = b1 * xi - a1 * yi + z2;
            z2 = b2 * xi - a2 * yi;
            y(i) = yi;
        }
        zf(s, 0) = z1;
        zf(s, 1) = z2;
    }

    return {y, zf};
}

ArrayXXr sosfilt_zi(const SOSFilter& sos) {
    int n_sections = static_cast<int>(sos.size());
    ArrayXXr zi(n_sections, 2);

    // Cumulative DC gain scaling (matching scipy's sosfilt_zi)
    // Each section's zi is scaled by the cumulative DC gain of all preceding sections
    Real scale = 1.0;

    for (int s = 0; s < n_sections; ++s) {
        Real b0 = sos[s][0], b1 = sos[s][1], b2 = sos[s][2];
        Real a1 = sos[s][4], a2 = sos[s][5];

        // Solve: [1+a1, -1] [z0]   [b1 - a1*b0]
        //        [a2,    1] [z1] = [b2 - a2*b0]
        Real det = (1.0 + a1) * 1.0 - (-1.0) * a2;  // = 1 + a1 + a2
        Real rhs0 = b1 - a1 * b0;
        Real rhs1 = b2 - a2 * b0;

        zi(s, 0) = scale * (rhs0 * 1.0 - (-1.0) * rhs1) / det;
        zi(s, 1) = scale * ((1.0 + a1) * rhs1 - a2 * rhs0) / det;

        // Accumulate DC gain: H(1) = sum(b) / sum(a) for this section
        Real b_sum = b0 + b1 + b2;
        Real a_sum = 1.0 + a1 + a2;  // a0 = 1
        if (std::abs(a_sum) > 1e-15) {
            scale *= b_sum / a_sum;
        }
    }

    return zi;
}

ArrayXr sosfiltfilt(const SOSFilter& sos, const ArrayXr& x) {
    int n_sections = static_cast<int>(sos.size());
    Eigen::Index n = x.size();

    // Compute initial conditions
    ArrayXXr zi = sosfilt_zi(sos);

    // Pad length (scipy default)
    int padlen = 3 * (2 * n_sections - 1);
    if (padlen >= n) {
        padlen = static_cast<int>(n) - 1;
    }

    // Odd-reflect pad: [2*x[0] - x[padlen:0:-1], x, 2*x[-1] - x[-2:-padlen-2:-1]]
    Eigen::Index padded_len = n + 2 * padlen;
    ArrayXr x_pad(padded_len);

    // Left padding: 2*x[0] - x[padlen], 2*x[0] - x[padlen-1], ..., 2*x[0] - x[1]
    for (int i = 0; i < padlen; ++i) {
        x_pad(i) = 2.0 * x(0) - x(padlen - i);
    }
    // Original signal
    x_pad.segment(padlen, n) = x;
    // Right padding: 2*x[-1] - x[-2], 2*x[-1] - x[-3], ..., 2*x[-1] - x[-padlen-1]
    for (int i = 0; i < padlen; ++i) {
        x_pad(padlen + n + i) = 2.0 * x(n - 1) - x(n - 2 - i);
    }

    // Forward pass: scale zi by x_pad[0]
    ArrayXXr zi_fwd = zi.colwise() * ArrayXr::Constant(n_sections, x_pad(0));
    auto [y_fwd, zf_fwd] = sosfilt(sos, x_pad, zi_fwd);

    // Reverse
    ArrayXr y_rev(padded_len);
    for (Eigen::Index i = 0; i < padded_len; ++i) {
        y_rev(i) = y_fwd(padded_len - 1 - i);
    }

    // Backward pass: scale zi by y_rev[0]
    ArrayXXr zi_bwd = zi.colwise() * ArrayXr::Constant(n_sections, y_rev(0));
    auto [y_bwd, zf_bwd] = sosfilt(sos, y_rev, zi_bwd);

    // Reverse again and strip padding
    ArrayXr result(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        result(i) = y_bwd(padded_len - 1 - padlen - i);
    }

    return result;
}

} // namespace filters
} // namespace librosa
