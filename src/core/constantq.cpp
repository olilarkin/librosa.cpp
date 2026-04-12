#include <librosa/core/constantq.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/core/pitch.hpp>
#include <librosa/filters.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include "internal/fft.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

namespace librosa {

namespace {

// ============================================================================
// Internal Helpers
// ============================================================================

/// Compute relative bandwidth for equal temperament
ArrayXr et_relative_bw(int bins_per_octave) {
    Real r = std::pow(2.0, 1.0 / bins_per_octave);
    ArrayXr alpha(1);
    alpha(0) = (r * r - 1.0) / (r * r + 1.0);
    return alpha;
}

/// Count factors of 2 in an integer
int num_two_factors(int x) {
    if (x <= 0) return 0;
    int count = 0;
    while (x % 2 == 0) {
        count++;
        x /= 2;
    }
    return count;
}

/// Pick the best available resampler for recursive CQT/VQT downsampling.
const char* cqt_resample_type() {
    return "kaiser_hq";
}

/// Compute the number of early downsampling operations
int early_downsample_count(Real nyquist, Real filter_cutoff, int hop_length, int n_octaves) {
    int ds1 = std::max(0, static_cast<int>(std::ceil(std::log2(nyquist / filter_cutoff)) - 1) - 1);
    int n2 = num_two_factors(hop_length);
    int ds2 = std::max(0, n2 - n_octaves + 1);
    return std::min(ds1, ds2);
}

/// Perform early downsampling on an audio signal
std::tuple<ArrayXr, Real, int> early_downsample(
    const ArrayXr& y, Real sr, int hop_length,
    int n_octaves, Real nyquist, Real filter_cutoff, bool scale) {

    int ds_count = early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves);

    if (ds_count > 0) {
        int ds_factor = 1 << ds_count;
        hop_length /= ds_factor;

        if (y.size() < ds_factor) {
            throw ParameterError("Input signal is too short for " +
                std::to_string(n_octaves) + "-octave CQT");
        }

        Real new_sr = sr / static_cast<Real>(ds_factor);
        ArrayXr y_ds = resample(y, static_cast<Real>(ds_factor), 1.0,
                                cqt_resample_type(), true, true);

        if (!scale) {
            y_ds *= std::sqrt(static_cast<Real>(ds_factor));
        }

        return {y_ds, new_sr, hop_length};
    }

    return {y, sr, hop_length};
}

/// Sparsify rows of a complex matrix by cumulative magnitude quantile
SparseMatrixXc sparsify_rows_complex(const ArrayXXc& x, Real quantile) {
    if (quantile <= 0) {
        // No sparsification — convert dense to sparse
        Eigen::Index rows = x.rows();
        Eigen::Index cols = x.cols();
        SparseMatrixXc sparse(rows, cols);

        std::vector<Eigen::Triplet<Complex>> triplets;
        triplets.reserve(rows * cols);
        for (Eigen::Index r = 0; r < rows; ++r) {
            for (Eigen::Index c = 0; c < cols; ++c) {
                if (x(r, c) != Complex(0.0, 0.0)) {
                    triplets.emplace_back(r, c, x(r, c));
                }
            }
        }
        sparse.setFromTriplets(triplets.begin(), triplets.end());
        return sparse;
    }

    Eigen::Index rows = x.rows();
    Eigen::Index cols = x.cols();

    std::vector<Eigen::Triplet<Complex>> triplets;

    for (Eigen::Index r = 0; r < rows; ++r) {
        // Compute magnitudes for this row
        std::vector<Real> mags(cols);
        for (Eigen::Index c = 0; c < cols; ++c) {
            mags[c] = std::abs(x(r, c));
        }

        // Sort magnitudes ascending
        std::vector<Real> sorted_mags = mags;
        std::sort(sorted_mags.begin(), sorted_mags.end());

        // Compute total magnitude (L1 norm, matching Python's sparsify_rows)
        Real total = 0.0;
        for (auto m : sorted_mags) {
            total += m;
        }

        if (total <= 0.0) continue;

        // Find threshold: cumulative normalized magnitude >= quantile
        Real cumsum = 0.0;
        Real threshold = 0.0;
        for (auto m : sorted_mags) {
            cumsum += m;
            if (cumsum / total >= quantile) {
                threshold = m;
                break;
            }
        }

        // Keep entries above threshold
        for (Eigen::Index c = 0; c < cols; ++c) {
            if (mags[c] >= threshold) {
                triplets.emplace_back(r, c, x(r, c));
            }
        }
    }

    SparseMatrixXc sparse(rows, cols);
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    return sparse;
}

/// Generate the frequency domain variable-Q filter basis
struct VQTFilterResult {
    SparseMatrixXc fft_basis;
    int n_fft;
    ArrayXr lengths;
};

VQTFilterResult vqt_filter_fft(
    Real sr, const ArrayXr& freqs, Real filter_scale,
    std::optional<Real> norm, Real sparsity,
    std::optional<int> hop_length,
    WindowType window,
    std::optional<Real> gamma,
    const ArrayXr& alpha) {

    auto [basis, lengths] = filters::wavelet(freqs, sr, window, filter_scale,
                                              true, norm, gamma, alpha);

    int n_fft = static_cast<int>(basis.cols());

    if (hop_length.has_value()) {
        int min_fft = static_cast<int>(std::pow(2.0, 1.0 + std::ceil(std::log2(hop_length.value()))));
        if (n_fft < min_fft) {
            n_fft = min_fft;
        }
    }

    // Re-normalize bases with respect to the FFT window length
    for (Eigen::Index i = 0; i < basis.rows(); ++i) {
        basis.row(i) *= lengths(i) / static_cast<Real>(n_fft);
    }

    // FFT and retain only the non-negative frequencies
    int n_out = n_fft / 2 + 1;
    ArrayXXc fft_basis(basis.rows(), n_out);
    {
        internal::ComplexFft cfft(n_fft);
        std::vector<Complex> in_buf(n_fft);
        std::vector<Complex> out_buf(n_fft);
        const Eigen::Index input_cols = basis.cols();
        const Eigen::Index copy_cols = std::min(input_cols, static_cast<Eigen::Index>(n_fft));
        for (Eigen::Index r = 0; r < basis.rows(); ++r) {
            std::fill(in_buf.begin(), in_buf.end(), Complex(0, 0));
            for (Eigen::Index c = 0; c < copy_cols; ++c) {
                in_buf[c] = basis(r, c);
            }
            cfft.forward(in_buf.data(), out_buf.data());
            for (int c = 0; c < n_out; ++c) {
                fft_basis(r, c) = out_buf[c];
            }
        }
    }

    // Sparsify
    SparseMatrixXc sparse_basis = sparsify_rows_complex(fft_basis, sparsity);

    return {sparse_basis, n_fft, lengths};
}

/// Compute the CQT filter response
ArrayXXc cqt_response(const ArrayXr& y, int n_fft, int hop_length,
                      const SparseMatrixXc& fft_basis, PadMode pad_mode) {
    // Compute STFT with rectangular window
    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt,
                      WindowType::Rectangular, true, pad_mode);

    Eigen::Index n_freq = D.rows();
    Eigen::Index n_frames = D.cols();

    // Convert D to matrix for sparse multiply
    // D is (n_freq, n_frames), fft_basis is (n_filters, n_freq)
    // Result = fft_basis * D → (n_filters, n_frames)
    MatrixXc D_mat = D.matrix();
    Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result_mat =
        fft_basis * D_mat;

    return result_mat.array();
}

/// Compute magnitude-only CQT response for pseudo_cqt
ArrayXXr cqt_response_abs(const ArrayXr& y, int n_fft, int hop_length,
                          const ArrayXXr& fft_basis_abs, PadMode pad_mode) {
    // Compute STFT with Hann window
    ArrayXXc D = stft(y, n_fft, hop_length, std::nullopt,
                      WindowType::Hann, true, pad_mode);

    // Take abs of STFT
    ArrayXXr D_abs = D.abs();

    Eigen::Index n_frames = D_abs.cols();

    // Matrix multiply: fft_basis_abs * D_abs → (n_filters, n_frames)
    MatrixXr basis_mat = fft_basis_abs.matrix();
    MatrixXr D_mat = D_abs.matrix();
    MatrixXr result_mat = basis_mat * D_mat;

    return result_mat.array();
}

/// Trim and stack a collection of CQT responses
ArrayXXc trim_stack(const std::vector<ArrayXXc>& vqt_resp, int n_bins) {
    // Find minimum number of time frames across octaves
    Eigen::Index max_col = vqt_resp[0].cols();
    for (const auto& c : vqt_resp) {
        max_col = std::min(max_col, c.cols());
    }

    ArrayXXc result(n_bins, max_col);
    result.setZero();

    // Copy per-octave data into output array
    int end = n_bins;
    for (const auto& c_i : vqt_resp) {
        int n_oct = static_cast<int>(c_i.rows());
        if (end < n_oct) {
            // Take the highest bins from c_i
            result.topRows(end) = c_i.bottomRows(end).leftCols(max_col);
        } else {
            result.block(end - n_oct, 0, n_oct, max_col) = c_i.leftCols(max_col);
        }
        end -= n_oct;
    }

    return result;
}

} // anonymous namespace

// ============================================================================
// VQT - Variable-Q Transform (core implementation)
// ============================================================================

ArrayXXc vqt(const ArrayXr& y, Real sr, int hop_length,
             std::optional<Real> fmin_opt, int n_bins,
             std::optional<Real> gamma,
             int bins_per_octave, std::optional<Real> tuning_opt,
             Real filter_scale, std::optional<Real> norm,
             Real sparsity, WindowType window,
             bool scale, PadMode pad_mode) {

    util::valid_audio(y);

    int n_octaves = static_cast<int>(std::ceil(static_cast<Real>(n_bins) / bins_per_octave));
    int n_filters = std::min(bins_per_octave, n_bins);

    Real fmin = fmin_opt.value_or(note_to_hz("C1"));

    Real tuning;
    if (tuning_opt.has_value()) {
        tuning = tuning_opt.value();
    } else {
        tuning = estimate_tuning(y, sr, 2048, 0.01, bins_per_octave);
    }

    // Apply tuning correction
    fmin *= std::pow(2.0, tuning / bins_per_octave);

    // Compute frequencies (equal temperament only)
    ArrayXr freqs = cqt_frequencies(n_bins, fmin, bins_per_octave);

    // Compute alpha (relative bandwidth)
    ArrayXr alpha;
    if (n_bins == 1) {
        alpha = et_relative_bw(bins_per_octave);
    } else {
        alpha = filters::relative_bandwidth(freqs);
    }

    // Compute filter lengths and cutoff frequency
    auto [lengths, filter_cutoff] = filters::wavelet_lengths(
        freqs, sr, window, filter_scale, gamma, alpha);

    Real nyquist = sr / 2.0;
    if (filter_cutoff > nyquist) {
        throw ParameterError("Wavelet basis would exceed Nyquist frequency. "
                             "Try reducing the number of frequency bins.");
    }

    // Early downsampling optimization
    auto [y_ds, sr_ds, hop_ds] = early_downsample(
        y, sr, hop_length, n_octaves, nyquist, filter_cutoff, scale);

    std::vector<ArrayXXc> vqt_resp;

    ArrayXr my_y = y_ds;
    Real my_sr = sr_ds;
    int my_hop = hop_ds;

    for (int i = 0; i < n_octaves; ++i) {
        // Slice out the current octave of filters (from highest to lowest)
        // Python uses negative slice indices which auto-clamp to 0.
        Eigen::Index sl_start, sl_end;
        if (i == 0) {
            sl_start = freqs.size() - n_filters;
            sl_end = freqs.size();
        } else {
            sl_start = std::max(Eigen::Index(0), freqs.size() - n_filters * (i + 1));
            sl_end = freqs.size() - n_filters * i;
        }

        ArrayXr freqs_oct = freqs.segment(sl_start, sl_end - sl_start);
        ArrayXr alpha_oct = alpha.segment(sl_start, sl_end - sl_start);

        auto filter_result = vqt_filter_fft(
            my_sr, freqs_oct, filter_scale, norm, sparsity,
            std::nullopt, window, gamma, alpha_oct);

        // Re-scale filters to compensate for downsampling
        filter_result.fft_basis *= std::sqrt(sr_ds / my_sr);

        // Compute VQT filter response
        vqt_resp.push_back(cqt_response(my_y, filter_result.n_fft, my_hop,
                                         filter_result.fft_basis, pad_mode));

        // Downsample for next octave
        if (my_hop % 2 == 0) {
            my_hop /= 2;
            my_sr /= 2.0;
            my_y = resample(my_y, 2.0, 1.0, cqt_resample_type(), true, true);
        }
    }

    ArrayXXc V = trim_stack(vqt_resp, n_bins);

    if (scale) {
        // Recompute lengths with the original (post-early-downsample) sr
        auto [scale_lengths, scale_cutoff] = filters::wavelet_lengths(
            freqs, sr_ds, window, filter_scale, gamma, alpha);

        for (Eigen::Index i = 0; i < V.rows(); ++i) {
            V.row(i) /= std::sqrt(scale_lengths(i));
        }
    }

    return V;
}

// ============================================================================
// CQT - Constant-Q Transform (wrapper around VQT with gamma=0)
// ============================================================================

ArrayXXc cqt(const ArrayXr& y, Real sr, int hop_length,
             std::optional<Real> fmin, int n_bins,
             int bins_per_octave, std::optional<Real> tuning,
             Real filter_scale, std::optional<Real> norm,
             Real sparsity, WindowType window,
             bool scale, PadMode pad_mode) {

    return vqt(y, sr, hop_length, fmin, n_bins,
               0.0,  // gamma = 0 for CQT
               bins_per_octave, tuning, filter_scale, norm,
               sparsity, window, scale, pad_mode);
}

// ============================================================================
// Pseudo CQT - Single FFT size, magnitude-only
// ============================================================================

ArrayXXr pseudo_cqt(const ArrayXr& y, Real sr, int hop_length,
                    std::optional<Real> fmin_opt, int n_bins,
                    int bins_per_octave, std::optional<Real> tuning_opt,
                    Real filter_scale, std::optional<Real> norm,
                    Real sparsity, WindowType window,
                    bool scale, PadMode pad_mode) {

    util::valid_audio(y);

    Real fmin = fmin_opt.value_or(note_to_hz("C1"));

    Real tuning;
    if (tuning_opt.has_value()) {
        tuning = tuning_opt.value();
    } else {
        tuning = estimate_tuning(y, sr, 2048, 0.01, bins_per_octave);
    }

    // Apply tuning correction
    fmin *= std::pow(2.0, tuning / bins_per_octave);

    ArrayXr freqs = cqt_frequencies(n_bins, fmin, bins_per_octave);

    ArrayXr alpha;
    if (n_bins == 1) {
        alpha = et_relative_bw(bins_per_octave);
    } else {
        alpha = filters::relative_bandwidth(freqs);
    }

    auto [lengths, pcqt_cutoff] = filters::wavelet_lengths(
        freqs, sr, window, filter_scale, std::optional<Real>(0.0), alpha);

    auto filter_result = vqt_filter_fft(
        sr, freqs, filter_scale, norm, sparsity,
        hop_length, window, std::optional<Real>(0.0), alpha);

    // Get dense absolute value of basis
    MatrixXc dense_basis = filter_result.fft_basis;
    ArrayXXr fft_basis_abs(dense_basis.rows(), dense_basis.cols());
    for (Eigen::Index r = 0; r < dense_basis.rows(); ++r) {
        for (Eigen::Index c = 0; c < dense_basis.cols(); ++c) {
            fft_basis_abs(r, c) = std::abs(dense_basis(r, c));
        }
    }

    ArrayXXr C = cqt_response_abs(y, filter_result.n_fft, hop_length,
                                   fft_basis_abs, pad_mode);

    if (scale) {
        C /= std::sqrt(static_cast<Real>(filter_result.n_fft));
    } else {
        for (Eigen::Index i = 0; i < C.rows(); ++i) {
            C.row(i) *= std::sqrt(lengths(i) / static_cast<Real>(filter_result.n_fft));
        }
    }

    return C;
}

// ============================================================================
// Hybrid CQT - Pseudo CQT for high freq + full CQT for low freq
// ============================================================================

ArrayXXr hybrid_cqt(const ArrayXr& y, Real sr, int hop_length,
                     std::optional<Real> fmin_opt, int n_bins,
                     int bins_per_octave, std::optional<Real> tuning_opt,
                     Real filter_scale, std::optional<Real> norm,
                     Real sparsity, WindowType window,
                     bool scale, PadMode pad_mode) {

    util::valid_audio(y);

    Real fmin = fmin_opt.value_or(note_to_hz("C1"));

    Real tuning;
    if (tuning_opt.has_value()) {
        tuning = tuning_opt.value();
    } else {
        tuning = estimate_tuning(y, sr, 2048, 0.01, bins_per_octave);
    }

    // Apply tuning correction
    fmin *= std::pow(2.0, tuning / bins_per_octave);

    // Get all CQT frequencies
    ArrayXr freqs = cqt_frequencies(n_bins, fmin, bins_per_octave);

    // Pre-compute alpha (relative bandwidth)
    ArrayXr alpha;
    if (n_bins == 1) {
        alpha = et_relative_bw(bins_per_octave);
    } else {
        alpha = filters::relative_bandwidth(freqs);
    }

    // Compute filter lengths
    auto [lengths, f_cutoff] = filters::wavelet_lengths(
        freqs, sr, window, filter_scale, std::optional<Real>(0.0), alpha);

    // Determine which filters to use with Pseudo CQT
    // These are the ones that fit within 2 hop lengths after padding
    int n_bins_pseudo = 0;
    for (Eigen::Index i = 0; i < lengths.size(); ++i) {
        Real next_pow2 = std::pow(2.0, std::ceil(std::log2(lengths(i))));
        if (next_pow2 < 2 * hop_length) {
            n_bins_pseudo++;
        }
    }

    int n_bins_full = n_bins - n_bins_pseudo;

    std::vector<ArrayXXc> cqt_resp;

    if (n_bins_pseudo > 0) {
        Real fmin_pseudo = freqs(n_bins - n_bins_pseudo);

        // Pass tuning=0.0 since fmin_pseudo is already tuning-corrected
        ArrayXXr pseudo = pseudo_cqt(y, sr, hop_length, fmin_pseudo,
                                      n_bins_pseudo, bins_per_octave, 0.0,
                                      filter_scale, norm, sparsity, window,
                                      scale, pad_mode);
        // Convert real to complex for trim_stack
        ArrayXXc pseudo_c(pseudo.rows(), pseudo.cols());
        for (Eigen::Index i = 0; i < pseudo.rows(); ++i) {
            for (Eigen::Index j = 0; j < pseudo.cols(); ++j) {
                pseudo_c(i, j) = Complex(pseudo(i, j), 0.0);
            }
        }
        cqt_resp.push_back(pseudo_c);
    }

    if (n_bins_full > 0) {
        // Pass tuning=0.0 since fmin is already tuning-corrected
        ArrayXXc full = cqt(y, sr, hop_length, fmin, n_bins_full,
                             bins_per_octave, 0.0, filter_scale, norm,
                             sparsity, window, scale, pad_mode);
        // Take absolute value, store as complex
        ArrayXXc full_abs(full.rows(), full.cols());
        for (Eigen::Index i = 0; i < full.rows(); ++i) {
            for (Eigen::Index j = 0; j < full.cols(); ++j) {
                full_abs(i, j) = Complex(std::abs(full(i, j)), 0.0);
            }
        }
        cqt_resp.push_back(full_abs);
    }

    ArrayXXc stacked = trim_stack(cqt_resp, n_bins);

    // Return real part (all values are real magnitudes stored as complex)
    return stacked.real();
}

// ============================================================================
// ICQT - Inverse Constant-Q Transform
// ============================================================================

ArrayXr icqt(const ArrayXXc& C, Real sr, int hop_length,
             std::optional<Real> fmin_opt, int bins_per_octave,
             Real tuning, Real filter_scale,
             std::optional<Real> norm, Real sparsity,
             WindowType window, bool scale,
             std::optional<int> length) {

    Real fmin = fmin_opt.value_or(note_to_hz("C1"));

    // Apply tuning correction
    fmin *= std::pow(2.0, tuning / bins_per_octave);

    ArrayXXc C_work = C;

    int n_bins = static_cast<int>(C_work.rows());
    int n_octaves = static_cast<int>(std::ceil(static_cast<Real>(n_bins) / bins_per_octave));

    ArrayXr freqs = cqt_frequencies(n_bins, fmin, bins_per_octave);

    ArrayXr alpha;
    if (n_bins == 1) {
        alpha = et_relative_bw(bins_per_octave);
    } else {
        alpha = filters::relative_bandwidth(freqs);
    }

    auto [lengths, f_cutoff] = filters::wavelet_lengths(
        freqs, sr, window, filter_scale, std::optional<Real>(0.0), alpha);

    if (length.has_value()) {
        int n_frames = static_cast<int>(
            std::ceil((length.value() + lengths.maxCoeff()) / static_cast<Real>(hop_length)));
        if (C_work.cols() > n_frames) {
            C_work = C_work.leftCols(n_frames);
        }
    }

    ArrayXr C_scale = lengths.sqrt();

    // Compute sample rates and hop lengths for each octave
    std::vector<Real> srs = {sr};
    std::vector<int> hops = {hop_length};

    for (int i = 0; i < n_octaves - 1; ++i) {
        if (hops[0] % 2 == 0) {
            srs.insert(srs.begin(), srs[0] * 0.5);
            hops.insert(hops.begin(), hops[0] / 2);
        } else {
            srs.insert(srs.begin(), srs[0]);
            hops.insert(hops.begin(), hops[0]);
        }
    }

    ArrayXr y_out;
    bool y_initialized = false;

    for (int i = 0; i < n_octaves; ++i) {
        Real my_sr = srs[i];
        int my_hop = hops[i];

        // How many filters in this octave?
        int n_filters = std::min(bins_per_octave, n_bins - bins_per_octave * i);
        Eigen::Index sl_start = bins_per_octave * i;

        // Get filter basis for this octave
        ArrayXr freqs_oct = freqs.segment(sl_start, n_filters);
        ArrayXr alpha_oct = alpha.segment(sl_start, n_filters);

        auto filter_result = vqt_filter_fft(
            my_sr, freqs_oct, filter_scale, norm, sparsity,
            std::nullopt, window, std::optional<Real>(0.0), alpha_oct);

        int n_fft = filter_result.n_fft;

        // Transpose the basis: inv_basis = fft_basis.conjugate().T
        // fft_basis is (n_filters, n_freq), inv_basis is (n_freq, n_filters)
        MatrixXc dense_basis = filter_result.fft_basis;
        MatrixXc inv_basis = dense_basis.conjugate().transpose();

        // Match librosa: normalize per filter/channel, not per FFT bin.
        ArrayXr freq_power(n_filters);
        for (int c = 0; c < n_filters; ++c) {
            Real power = 0.0;
            for (int f = 0; f < static_cast<int>(inv_basis.rows()); ++f) {
                power += std::norm(inv_basis(f, c));
            }
            freq_power(c) = (power > 0.0) ? 1.0 / power : 0.0;
        }

        // Compensate for length normalization in forward transform
        ArrayXr lengths_oct = lengths.segment(sl_start, n_filters);
        for (int c = 0; c < n_filters; ++c) {
            freq_power(c) *= static_cast<Real>(n_fft) / lengths_oct(c);
        }
        ArrayXr C_scale_oct = C_scale.segment(sl_start, n_filters);

        // D_oct[f, t] = sum_c inv_basis[f,c] * scale[c] * C[sl_start+c, t]
        ArrayXXc C_oct = C_work.block(sl_start, 0, n_filters, C_work.cols());

        ArrayXXc C_scaled(n_filters, C_work.cols());
        for (int c = 0; c < n_filters; ++c) {
            Real s = freq_power(c);
            if (scale) {
                s *= C_scale_oct(c);
            }
            C_scaled.row(c) = C_oct.row(c) * s;
        }

        // D_oct = inv_basis * C_scaled → (n_freq, n_frames)
        MatrixXc D_oct_mat = inv_basis * C_scaled.matrix();

        ArrayXXc D_oct = D_oct_mat.array();

        // ISTFT with ones window
        ArrayXr y_oct = istft(D_oct, my_hop, std::nullopt, std::nullopt,
                              WindowType::Rectangular, true, std::nullopt);

        // Resample to full rate if needed
        if (my_sr != sr) {
            Real ratio = sr / my_sr;
            y_oct = resample(y_oct, 1.0, ratio, cqt_resample_type(), false, false);
        }

        if (!y_initialized) {
            y_out = y_oct;
            y_initialized = true;
        } else {
            Eigen::Index min_len = std::min(y_out.size(), y_oct.size());
            y_out.head(min_len) += y_oct.head(min_len);
            if (y_oct.size() > y_out.size()) {
                ArrayXr extended = ArrayXr::Zero(y_oct.size());
                extended.head(y_out.size()) = y_out;
                extended.tail(y_oct.size() - y_out.size()) = y_oct.tail(y_oct.size() - y_out.size());
                y_out = extended;
            }
        }
    }

    if (length.has_value()) {
        y_out = util::fix_length(y_out, length.value());
    }

    return y_out;
}

// ============================================================================
// Griffin-Lim CQT
// ============================================================================

ArrayXr griffinlim_cqt(const ArrayXXr& C, int n_iter, Real sr, int hop_length,
                       std::optional<Real> fmin, int bins_per_octave,
                       Real tuning, Real filter_scale,
                       std::optional<Real> norm, Real sparsity,
                       WindowType window, bool scale,
                       PadMode pad_mode, std::optional<int> length,
                       Real momentum, const std::string& init_phase,
                       std::optional<unsigned int> random_state) {

    if (momentum < 0) {
        throw ParameterError("griffinlim_cqt() called with momentum < 0");
    }

    Real eps = util::tiny<Real>();

    // Initialize phase
    ArrayXXc angles(C.rows(), C.cols());
    if (init_phase == "random") {
        std::mt19937 gen(random_state.value_or(std::random_device{}()));
        std::uniform_real_distribution<Real> dist(-constants::PI, constants::PI);
        for (Eigen::Index i = 0; i < C.rows(); ++i) {
            for (Eigen::Index j = 0; j < C.cols(); ++j) {
                angles(i, j) = std::polar(1.0, dist(gen));
            }
        }
    } else {
        angles.setOnes();
    }

    ArrayXXc rebuilt = ArrayXXc::Zero(C.rows(), C.cols());

    for (int iter = 0; iter < n_iter; ++iter) {
        ArrayXXc tprev = rebuilt;

        // Build complex spectrogram from magnitude and current phase
        ArrayXXc C_complex(C.rows(), C.cols());
        for (Eigen::Index i = 0; i < C.rows(); ++i) {
            for (Eigen::Index j = 0; j < C.cols(); ++j) {
                C_complex(i, j) = C(i, j) * angles(i, j);
            }
        }

        // Invert
        ArrayXr inverse = icqt(C_complex, sr, hop_length, fmin,
                                bins_per_octave, tuning, filter_scale,
                                norm, sparsity, window, scale, length);

        // Forward transform
        rebuilt = cqt(inverse, sr, hop_length, fmin,
                      static_cast<int>(C.rows()), bins_per_octave,
                      tuning, filter_scale, norm, sparsity,
                      window, scale, pad_mode);

        // Update phases with momentum
        ArrayXXc new_angles = rebuilt - (momentum / (1.0 + momentum)) * tprev;
        for (Eigen::Index i = 0; i < C.rows(); ++i) {
            for (Eigen::Index j = 0; j < C.cols(); ++j) {
                Real mag = std::abs(new_angles(i, j));
                if (mag > eps) {
                    angles(i, j) = new_angles(i, j) / mag;
                } else {
                    angles(i, j) = Complex(1.0, 0.0);
                }
            }
        }
    }

    // Final inversion
    ArrayXXc C_final(C.rows(), C.cols());
    for (Eigen::Index i = 0; i < C.rows(); ++i) {
        for (Eigen::Index j = 0; j < C.cols(); ++j) {
            C_final(i, j) = C(i, j) * angles(i, j);
        }
    }

    return icqt(C_final, sr, hop_length, fmin, bins_per_octave,
                tuning, filter_scale, norm, sparsity, window, scale, length);
}

} // namespace librosa
