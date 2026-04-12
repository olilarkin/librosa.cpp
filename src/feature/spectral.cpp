#include <librosa/feature/spectral.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/core/constantq.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/pitch.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/filters.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include "../internal/dct.hpp"
#include "../internal/convolve2d.hpp"
#include <cmath>
#include <algorithm>

namespace librosa {
namespace feature {

// Use shared DCT implementations
using internal::dct_ii;
using internal::dct_iii;

namespace {

ArrayXXr matrix_product(const ArrayXXr& lhs, const ArrayXXr& rhs) {
    if (lhs.cols() != rhs.rows()) {
        throw ParameterError("Matrix dimensions are incompatible for multiplication");
    }

#ifdef __EMSCRIPTEN__
    ArrayXXr result = ArrayXXr::Zero(lhs.rows(), rhs.cols());
    for (Eigen::Index r = 0; r < lhs.rows(); ++r) {
        for (Eigen::Index k = 0; k < lhs.cols(); ++k) {
            const Real left = lhs(r, k);
            if (left == 0.0) continue;
            for (Eigen::Index c = 0; c < rhs.cols(); ++c) {
                result(r, c) += left * rhs(k, c);
            }
        }
    }
    return result;
#else
    MatrixXr result = lhs.matrix() * rhs.matrix();
    return result.array();
#endif
}

// Compute spectrogram from audio or use provided one
std::pair<ArrayXXr, int> get_spectrogram(
    const ArrayXr* y,
    const ArrayXXr* S,
    int n_fft,
    int hop_length,
    std::optional<int> win_length,
    WindowType window,
    bool center,
    Real power = 1.0) {

    if (S != nullptr) {
        // Use provided spectrogram
        return {*S, n_fft};
    }

    if (y == nullptr) {
        throw ParameterError("Either y or S must be provided");
    }

    // Compute STFT
    ArrayXXc D = stft(*y, n_fft, hop_length, win_length, window, center);
    ArrayXXr mag = magnitude(D);

    if (power != 1.0) {
        mag = mag.pow(power);
    }

    return {mag, n_fft};
}

} // anonymous namespace

// ============================================================================
// Mel Spectrogram
// ============================================================================

ArrayXXr melspectrogram(
    const ArrayXr& y,
    Real sr,
    int n_fft,
    int hop_length,
    std::optional<int> win_length,
    WindowType window,
    bool center,
    PadMode pad_mode,
    Real power,
    int n_mels,
    Real fmin,
    std::optional<Real> fmax,
    bool htk,
    bool norm_slaney) {

    // Compute STFT
    ArrayXXc D = stft(y, n_fft, hop_length, win_length, window, center);
    ArrayXXr S = magnitude(D).pow(power);

    return melspectrogram(S, sr, n_fft, n_mels, fmin, fmax, htk, norm_slaney);
}

ArrayXXr melspectrogram(
    const ArrayXXr& S,
    Real sr,
    int n_fft,
    int n_mels,
    Real fmin,
    std::optional<Real> fmax,
    bool htk,
    bool norm_slaney) {

    Real f_max = fmax.value_or(sr / 2);

    // Build mel filter bank
    filters::MelNorm norm = norm_slaney ? filters::MelNorm::Slaney : filters::MelNorm::None;
    ArrayXXr mel_basis = filters::mel(sr, n_fft, n_mels, fmin, f_max, htk, norm);

    // Apply mel filterbank: result = mel_basis @ S
    // mel_basis: (n_mels, n_fft/2+1), S: (n_fft/2+1, t)
    return matrix_product(mel_basis, S);
}

// ============================================================================
// MFCC
// ============================================================================

ArrayXXr mfcc(
    const ArrayXr& y,
    Real sr,
    int n_mfcc,
    int dct_type,
    bool norm_ortho,
    Real lifter,
    int n_fft,
    int hop_length,
    int n_mels,
    Real fmin,
    std::optional<Real> fmax,
    bool htk) {

    // Compute mel spectrogram
    ArrayXXr S_mel = melspectrogram(y, sr, n_fft, hop_length, std::nullopt,
                                    WindowType::Hann, true, PadMode::Constant,
                                    2.0, n_mels, fmin, fmax, htk, true);

    // Convert to log scale (dB)
    ArrayXXr S_db = power_to_db(S_mel, 1.0, 1e-10, 80.0);

    return mfcc(S_db, n_mfcc, dct_type, norm_ortho, lifter);
}

ArrayXXr mfcc(
    const ArrayXXr& S,
    int n_mfcc,
    int dct_type,
    bool norm_ortho,
    Real lifter) {

    if (lifter < 0) {
        throw ParameterError("MFCC lifter must be a non-negative number");
    }

    // Apply DCT
    ArrayXXr M;
    if (dct_type == 2) {
        M = dct_ii(S, n_mfcc, norm_ortho);
    } else if (dct_type == 3) {
        M = dct_iii(S, n_mfcc, norm_ortho);
    } else {
        throw ParameterError("dct_type must be 2 or 3");
    }

    // Apply liftering
    if (lifter > 0) {
        ArrayXr LI(n_mfcc);
        for (int n = 0; n < n_mfcc; ++n) {
            LI(n) = std::sin(M_PI * (n + 1) / lifter);
        }

        for (Eigen::Index t = 0; t < M.cols(); ++t) {
            for (int n = 0; n < n_mfcc; ++n) {
                M(n, t) *= 1 + (lifter / 2) * LI(n);
            }
        }
    }

    return M;
}

// ============================================================================
// Chroma
// ============================================================================

ArrayXXr chroma_stft(
    const ArrayXr& y,
    Real sr,
    int n_fft,
    int hop_length,
    int n_chroma,
    std::optional<Real> tuning,
    Real norm,
    WindowType window,
    bool center) {

    // Compute STFT
    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt, window, center);
    ArrayXXr S = magnitude(D).square();  // Power spectrogram

    return chroma_stft(S, sr, n_fft, n_chroma, tuning, norm);
}

ArrayXXr chroma_stft(
    const ArrayXXr& S,
    Real sr,
    int n_fft,
    int n_chroma,
    std::optional<Real> tuning,
    Real norm) {

    // Estimate tuning if not provided
    Real tune = tuning.value_or(estimate_tuning(S, sr, n_fft, 0.01, n_chroma));

    // Get chroma filter bank
    ArrayXXr chromafb = filters::chroma(sr, n_fft, n_chroma, tune);

    // Compute raw chroma: chromafb @ S
    // chromafb: (n_chroma, n_fft/2+1), S: (n_fft/2+1, t)
    ArrayXXr raw_chroma = matrix_product(chromafb, S);

    // Normalize
    ArrayXXr result = util::normalize(raw_chroma, norm, 0);

    return result;
}

// ============================================================================
// Chroma CQT
// ============================================================================

ArrayXXr chroma_cqt(
    const ArrayXr& y,
    Real sr,
    int hop_length,
    std::optional<Real> fmin,
    Real norm,
    Real threshold,
    std::optional<Real> tuning,
    int n_chroma,
    int n_octaves,
    int bins_per_octave) {

    if (bins_per_octave % n_chroma != 0) {
        throw ParameterError("bins_per_octave must be an integer multiple of n_chroma");
    }

    int n_bins = n_octaves * bins_per_octave;
    ArrayXXc C_complex = cqt(y, sr, hop_length, fmin, n_bins, bins_per_octave, tuning);
    ArrayXXr C = magnitude(C_complex);

    return chroma_cqt(C, fmin, norm, threshold, n_chroma, bins_per_octave);
}

ArrayXXr chroma_cqt(
    const ArrayXXr& C,
    std::optional<Real> fmin,
    Real norm,
    Real threshold,
    int n_chroma,
    int bins_per_octave) {

    // Build CQ-to-chroma mapping
    ArrayXXr cq_to_chr = filters::cq_to_chroma(
        static_cast<int>(C.rows()), bins_per_octave, n_chroma, fmin);

    // Matrix multiply: chroma = cq_to_chr @ C
    ArrayXXr chroma = matrix_product(cq_to_chr, C);

    // Apply threshold
    ArrayXXr result = chroma;
    if (threshold > 0.0) {
        result = (result < threshold).select(0.0, result);
    }

    // Normalize (negative norm means no normalization)
    if (norm >= 0.0 || std::isinf(norm)) {
        result = util::normalize(result, norm, 0);
    }

    return result;
}

// ============================================================================
// Chroma CENS
// ============================================================================

ArrayXXr chroma_cens(
    const ArrayXr& y,
    Real sr,
    int hop_length,
    std::optional<Real> fmin,
    std::optional<Real> tuning,
    int n_chroma,
    int n_octaves,
    int bins_per_octave,
    Real norm,
    int win_len_smooth) {

    if (win_len_smooth < 0) {
        throw ParameterError("win_len_smooth must be a non-negative integer");
    }

    if (bins_per_octave % n_chroma != 0) {
        throw ParameterError("bins_per_octave must be an integer multiple of n_chroma");
    }

    // Compute CQT
    int n_bins = n_octaves * bins_per_octave;
    ArrayXXc C_complex = cqt(y, sr, hop_length, fmin, n_bins, bins_per_octave, tuning);
    ArrayXXr C = magnitude(C_complex);

    // Map to chroma (no threshold, no normalization)
    ArrayXXr cq_to_chr = filters::cq_to_chroma(
        static_cast<int>(C.rows()), bins_per_octave, n_chroma, fmin);
    ArrayXXr chroma = matrix_product(cq_to_chr, C);

    // L1-normalize
    chroma = util::normalize(chroma, 1.0, 0);

    // Quantize amplitudes
    const Real quant_steps[] = {0.4, 0.2, 0.1, 0.05};
    const Real quant_weights[] = {0.25, 0.25, 0.25, 0.25};

    ArrayXXr chroma_quant = ArrayXXr::Zero(chroma.rows(), chroma.cols());
    for (int q = 0; q < 4; ++q) {
        chroma_quant += (chroma > quant_steps[q]).template cast<Real>() * quant_weights[q];
    }

    // Temporal smoothing
    if (win_len_smooth > 0) {
        int win_len = win_len_smooth + 2;
        // Get Hann window with fftbins=false (symmetric)
        ArrayXr win = get_window(WindowType::Hann, win_len, false);
        Real win_sum = win.sum();
        win /= win_sum;

        // Create 2D kernel: (1, win_len) for row-wise convolution
        ArrayXXr kernel(1, win_len);
        kernel.row(0) = win;

        // Apply convolution with constant (zero) padding
        chroma_quant = internal::convolve2d(chroma_quant, kernel, "constant");
    }

    // L2-normalize (or specified norm)
    ArrayXXr result = util::normalize(chroma_quant, norm, 0);

    return result;
}

// ============================================================================
// Chroma VQT
// ============================================================================

ArrayXXr chroma_vqt(
    const ArrayXr& y,
    Real sr,
    int hop_length,
    std::optional<Real> fmin,
    Real norm,
    Real threshold,
    int n_octaves,
    int bins_per_octave,
    Real gamma) {

    int n_bins = n_octaves * bins_per_octave;
    std::optional<Real> gamma_opt = (gamma == 0.0) ? std::nullopt : std::optional<Real>(gamma);
    ArrayXXc V_complex = vqt(y, sr, hop_length, fmin, n_bins, gamma_opt, bins_per_octave);
    ArrayXXr V = magnitude(V_complex);

    return chroma_vqt(V, fmin, norm, threshold, bins_per_octave);
}

ArrayXXr chroma_vqt(
    const ArrayXXr& V,
    std::optional<Real> fmin,
    Real norm,
    Real threshold,
    int bins_per_octave) {

    // For VQT chroma, n_chroma = bins_per_octave (no aggregation)
    ArrayXXr vq_to_chr = filters::cq_to_chroma(
        static_cast<int>(V.rows()), bins_per_octave, bins_per_octave, fmin);

    ArrayXXr chroma = matrix_product(vq_to_chr, V);

    ArrayXXr result = chroma;
    if (threshold > 0.0) {
        result = (result < threshold).select(0.0, result);
    }

    if (norm >= 0.0 || std::isinf(norm)) {
        result = util::normalize(result, norm, 0);
    }

    return result;
}

// ============================================================================
// Spectral Centroid
// ============================================================================

ArrayXXr spectral_centroid(
    const ArrayXr& y,
    Real sr,
    int n_fft,
    int hop_length,
    WindowType window,
    bool center) {

    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt, window, center);
    ArrayXXr S = magnitude(D);

    return spectral_centroid(S, sr, n_fft, nullptr);
}

ArrayXXr spectral_centroid(
    const ArrayXXr& S,
    Real sr,
    int n_fft,
    const ArrayXr* freq_ptr) {

    // Check input
    if ((S < 0).any()) {
        throw ParameterError("Spectral centroid is only defined with non-negative energies");
    }

    // Get frequencies
    ArrayXr freq;
    if (freq_ptr != nullptr) {
        freq = *freq_ptr;
    } else {
        freq = fft_frequencies(sr, n_fft);
    }

    Eigen::Index n_frames = S.cols();
    ArrayXXr centroid(1, n_frames);

    // Normalize each frame and compute centroid
    for (Eigen::Index t = 0; t < n_frames; ++t) {
        Real total = S.col(t).sum();
        if (total > util::tiny(total)) {
            centroid(0, t) = (freq * S.col(t)).sum() / total;
        } else {
            centroid(0, t) = 0;
        }
    }

    return centroid;
}

// ============================================================================
// Spectral Bandwidth
// ============================================================================

ArrayXXr spectral_bandwidth(
    const ArrayXr& y,
    Real sr,
    int n_fft,
    int hop_length,
    WindowType window,
    bool center,
    Real p,
    bool norm) {

    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt, window, center);
    ArrayXXr S = magnitude(D);

    return spectral_bandwidth(S, sr, n_fft, nullptr, nullptr, p, norm);
}

ArrayXXr spectral_bandwidth(
    const ArrayXXr& S,
    Real sr,
    int n_fft,
    const ArrayXXr* centroid_ptr,
    const ArrayXr* freq_ptr,
    Real p,
    bool norm_energy) {

    // Check input
    if ((S < 0).any()) {
        throw ParameterError("Spectral bandwidth is only defined with non-negative energies");
    }

    // Get frequencies
    ArrayXr freq;
    if (freq_ptr != nullptr) {
        freq = *freq_ptr;
    } else {
        freq = fft_frequencies(sr, n_fft);
    }

    // Compute centroid if not provided
    ArrayXXr centroid;
    if (centroid_ptr != nullptr) {
        centroid = *centroid_ptr;
    } else {
        centroid = spectral_centroid(S, sr, n_fft, &freq);
    }

    Eigen::Index n_frames = S.cols();
    ArrayXXr bandwidth(1, n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        // Compute deviation from centroid
        ArrayXr deviation = (freq - centroid(0, t)).abs();

        // Normalize if requested
        ArrayXr weights = S.col(t);
        if (norm_energy) {
            Real total = weights.sum();
            if (total > util::tiny(total)) {
                weights /= total;
            }
        }

        // Compute bandwidth
        bandwidth(0, t) = std::pow((weights * deviation.pow(p)).sum(), 1.0 / p);
    }

    return bandwidth;
}

// ============================================================================
// Spectral Rolloff
// ============================================================================

ArrayXXr spectral_rolloff(
    const ArrayXr& y,
    Real sr,
    int n_fft,
    int hop_length,
    WindowType window,
    bool center,
    Real roll_percent) {

    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt, window, center);
    ArrayXXr S = magnitude(D);

    return spectral_rolloff(S, sr, n_fft, nullptr, roll_percent);
}

ArrayXXr spectral_rolloff(
    const ArrayXXr& S,
    Real sr,
    int n_fft,
    const ArrayXr* freq_ptr,
    Real roll_percent) {

    if (roll_percent <= 0 || roll_percent >= 1) {
        throw ParameterError("roll_percent must lie in the range (0, 1)");
    }

    // Check input
    if ((S < 0).any()) {
        throw ParameterError("Spectral rolloff is only defined with non-negative energies");
    }

    // Get frequencies
    ArrayXr freq;
    if (freq_ptr != nullptr) {
        freq = *freq_ptr;
    } else {
        freq = fft_frequencies(sr, n_fft);
    }

    Eigen::Index n_freq = S.rows();
    Eigen::Index n_frames = S.cols();
    ArrayXXr rolloff(1, n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        // Compute cumulative energy
        Real total_energy = S.col(t).sum();
        Real threshold = roll_percent * total_energy;

        Real cumsum = 0;
        rolloff(0, t) = freq(n_freq - 1);  // Default to max frequency

        for (Eigen::Index f = 0; f < n_freq; ++f) {
            cumsum += S(f, t);
            if (cumsum >= threshold) {
                rolloff(0, t) = freq(f);
                break;
            }
        }
    }

    return rolloff;
}

// ============================================================================
// Spectral Flatness
// ============================================================================

ArrayXXr spectral_flatness(
    const ArrayXr& y,
    int n_fft,
    int hop_length,
    WindowType window,
    bool center,
    Real amin,
    Real power) {

    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt, window, center);
    ArrayXXr S = magnitude(D);

    return spectral_flatness(S, amin, power);
}

ArrayXXr spectral_flatness(
    const ArrayXXr& S,
    Real amin,
    Real power) {

    if (amin <= 0) {
        throw ParameterError("amin must be strictly positive");
    }

    // Check input
    if ((S < 0).any()) {
        throw ParameterError("Spectral flatness is only defined with non-negative energies");
    }

    // Apply power and threshold
    ArrayXXr S_thresh = S.pow(power).max(amin);

    Eigen::Index n_freq = S_thresh.rows();
    Eigen::Index n_frames = S_thresh.cols();
    ArrayXXr flatness(1, n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        // Geometric mean
        Real log_sum = S_thresh.col(t).log().sum();
        Real gmean = std::exp(log_sum / n_freq);

        // Arithmetic mean
        Real amean = S_thresh.col(t).mean();

        flatness(0, t) = gmean / amean;
    }

    return flatness;
}

// ============================================================================
// Spectral Contrast
// ============================================================================

ArrayXXr spectral_contrast(
    const ArrayXr& y,
    Real sr,
    int n_fft,
    int hop_length,
    WindowType window,
    bool center,
    Real fmin,
    int n_bands,
    Real quantile,
    bool linear) {

    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt, window, center);
    ArrayXXr S = magnitude(D);

    return spectral_contrast(S, sr, n_fft, nullptr, fmin, n_bands, quantile, linear);
}

ArrayXXr spectral_contrast(
    const ArrayXXr& S,
    Real sr,
    int n_fft,
    const ArrayXr* freq_ptr,
    Real fmin,
    int n_bands,
    Real quantile,
    bool linear) {

    if (n_bands < 1) {
        throw ParameterError("n_bands must be a positive integer");
    }
    if (quantile <= 0 || quantile >= 1) {
        throw ParameterError("quantile must lie in the range (0, 1)");
    }
    if (fmin <= 0) {
        throw ParameterError("fmin must be a positive number");
    }

    // Get frequencies
    ArrayXr freq;
    if (freq_ptr != nullptr) {
        freq = *freq_ptr;
    } else {
        freq = fft_frequencies(sr, n_fft);
    }

    // Create octave bands
    std::vector<Real> octa(n_bands + 2);
    octa[0] = 0;
    for (int i = 0; i <= n_bands; ++i) {
        octa[i + 1] = fmin * std::pow(2.0, i);
    }

    // Check frequency range
    if (octa[n_bands] >= sr / 2) {
        throw ParameterError("Frequency band exceeds Nyquist. Reduce either fmin or n_bands.");
    }

    Eigen::Index n_frames = S.cols();
    ArrayXXr peak_arr(n_bands + 1, n_frames);
    ArrayXXr valley_arr(n_bands + 1, n_frames);

    for (int k = 0; k <= n_bands; ++k) {
        Real f_low = octa[k];
        Real f_high = octa[k + 1];

        // Find bins in this band
        std::vector<Eigen::Index> band_indices;
        for (Eigen::Index f = 0; f < freq.size(); ++f) {
            if (freq(f) >= f_low && freq(f) <= f_high) {
                band_indices.push_back(f);
            }
        }

        // Add neighboring bin at lower edge (except for first band)
        if (k > 0 && !band_indices.empty() && band_indices[0] > 0) {
            band_indices.insert(band_indices.begin(), band_indices[0] - 1);
        }

        // For last band, include all remaining bins
        if (k == n_bands && !band_indices.empty()) {
            for (Eigen::Index f = band_indices.back() + 1; f < freq.size(); ++f) {
                band_indices.push_back(f);
            }
        }

        // Compute quantile index BEFORE trimming (matching Python behavior)
        int n_bins_for_quantile = static_cast<int>(band_indices.size());

        // Remove last bin for non-final bands
        if (k < n_bands && band_indices.size() > 1) {
            band_indices.pop_back();
        }

        int n_bins = static_cast<int>(band_indices.size());
        int idx = std::max(1, static_cast<int>(std::round(quantile * n_bins_for_quantile)));

        for (Eigen::Index t = 0; t < n_frames; ++t) {
            // Extract sub-band values
            std::vector<Real> sub_band;
            for (Eigen::Index i : band_indices) {
                sub_band.push_back(S(i, t));
            }

            // Sort
            std::sort(sub_band.begin(), sub_band.end());

            // Compute valley (mean of lowest quantile)
            Real valley = 0;
            for (int i = 0; i < idx; ++i) {
                valley += sub_band[i];
            }
            valley /= idx;

            // Compute peak (mean of highest quantile)
            Real peak = 0;
            for (int i = n_bins - idx; i < n_bins; ++i) {
                peak += sub_band[i];
            }
            peak /= idx;

            peak_arr(k, t) = peak;
            valley_arr(k, t) = valley;
        }
    }

    // Compute contrast: Python applies power_to_db on the full arrays
    // (with top_db=80.0 default), so we must do the same to match behavior.
    if (linear) {
        return peak_arr - valley_arr;
    } else {
        return power_to_db(peak_arr) - power_to_db(valley_arr);
    }
}

// ============================================================================
// RMS
// ============================================================================

ArrayXXr rms(
    const ArrayXr& y,
    int frame_length,
    int hop_length,
    bool center) {

    ArrayXr y_padded;
    if (center) {
        int pad = frame_length / 2;
        y_padded.resize(y.size() + 2 * pad);
        y_padded.head(pad).setZero();
        y_padded.segment(pad, y.size()) = y;
        y_padded.tail(pad).setZero();
    } else {
        y_padded = y;
    }

    // Frame the signal
    ArrayXXr frames = util::frame(y_padded, frame_length, hop_length);

    // Compute RMS for each frame
    Eigen::Index n_frames = frames.cols();
    ArrayXXr result(1, n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        result(0, t) = std::sqrt(frames.col(t).square().mean());
    }

    return result;
}

ArrayXXr rms(
    const ArrayXXr& S,
    int frame_length) {

    // Check frame length consistency
    Eigen::Index expected_bins = frame_length / 2 + 1;
    if (S.rows() != expected_bins) {
        throw ParameterError("S.rows() does not match expected frame_length");
    }

    // Compute RMS from power spectrogram
    Eigen::Index n_frames = S.cols();
    ArrayXXr result(1, n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        result(0, t) = std::sqrt(S.col(t).square().mean());
    }

    return result;
}

// ============================================================================
// Zero Crossing Rate
// ============================================================================

ArrayXXr zero_crossing_rate(
    const ArrayXr& y,
    int frame_length,
    int hop_length,
    bool center,
    Real threshold) {

    ArrayXr y_padded;
    if (center) {
        int pad = frame_length / 2;
        y_padded.resize(y.size() + 2 * pad);
        y_padded.head(pad).setZero();
        y_padded.segment(pad, y.size()) = y;
        y_padded.tail(pad).setZero();
    } else {
        y_padded = y;
    }

    // Frame the signal
    ArrayXXr frames = util::frame(y_padded, frame_length, hop_length);

    Eigen::Index n_frames = frames.cols();
    ArrayXXr result(1, n_frames);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        ArrayXr frame = frames.col(t);
        auto crossings = zero_crossings(frame, threshold, std::nullopt, false);
        result(0, t) = static_cast<Real>(crossings.count()) / (frame_length - 1);
    }

    return result;
}

// ============================================================================
// Polynomial Features
// ============================================================================

ArrayXXr poly_features(
    const ArrayXXr& S,
    Real sr,
    int n_fft,
    int order,
    const ArrayXr* freq_ptr) {

    // Get frequencies
    ArrayXr freq;
    if (freq_ptr != nullptr) {
        freq = *freq_ptr;
    } else {
        freq = fft_frequencies(sr, n_fft);
    }

    Eigen::Index n_freq = S.rows();
    Eigen::Index n_frames = S.cols();

    // Compute polynomial coefficients for each frame using least-squares
    // matching numpy.polyfit(freq, S[:, t], order) behavior
    ArrayXXr coeffs(order + 1, n_frames);

    // Build Vandermonde matrix once (frequencies are constant across frames)
    // np.polyfit uses a Vandermonde matrix with raw frequencies (no normalization)
    MatrixXr V(n_freq, order + 1);
    for (Eigen::Index f = 0; f < n_freq; ++f) {
        for (int p = 0; p <= order; ++p) {
            V(f, p) = std::pow(freq(f), order - p);
        }
    }

    // Use QR decomposition for numerical stability (matching numpy.polyfit)
    auto qr = V.householderQr();

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        VectorXr y = S.col(t).matrix();
        coeffs.col(t) = qr.solve(y).array();
    }

    return coeffs;
}

ArrayXXr poly_features(
    const ArrayXr& y,
    Real sr,
    int n_fft,
    int hop_length,
    int order) {

    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt, WindowType::Hann, true);
    ArrayXXr S = magnitude(D);

    return poly_features(S, sr, n_fft, order, nullptr);
}

// ============================================================================
// Tonnetz
// ============================================================================

ArrayXXr tonnetz(
    const ArrayXr& y,
    Real sr,
    const ArrayXXr* chroma_ptr) {

    ArrayXXr chroma_data;
    if (chroma_ptr != nullptr) {
        chroma_data = *chroma_ptr;
    } else {
        // Compute chroma from CQT (matching Python default)
        chroma_data = chroma_cqt(y, sr);
    }

    return tonnetz(chroma_data);
}

ArrayXXr tonnetz(const ArrayXXr& chroma) {
    Eigen::Index n_chroma = chroma.rows();
    Eigen::Index n_frames = chroma.cols();

    // Generate transformation matrix
    ArrayXr dim_map(n_chroma);
    for (Eigen::Index i = 0; i < n_chroma; ++i) {
        dim_map(i) = 12.0 * i / n_chroma;
    }

    // Scale factors: fifth (7/6), minor (3/2), major (2/3)
    Real scales[6] = {7.0/6, 7.0/6, 3.0/2, 3.0/2, 2.0/3, 2.0/3};
    Real radii[6] = {1.0, 1.0, 1.0, 1.0, 0.5, 0.5};

    // Build phi matrix (6 x n_chroma)
    ArrayXXr phi(6, n_chroma);
    for (int i = 0; i < 6; ++i) {
        for (Eigen::Index c = 0; c < n_chroma; ++c) {
            Real v = scales[i] * dim_map(c);
            if (i % 2 == 0) {
                v -= 0.5;
            }
            phi(i, c) = radii[i] * std::cos(M_PI * v);
        }
    }

    // Normalize chroma
    ArrayXXr chroma_norm = util::normalize(chroma, 1, 0);

    // Compute tonnetz: phi @ chroma_norm
    return matrix_product(phi, chroma_norm);
}

} // namespace feature
} // namespace librosa
