#include "librosa/core/spectrum.hpp"
#include "librosa/core/audio.hpp"
#include "librosa/core/convert.hpp"
#include "librosa/filters.hpp"
#include "librosa/util/utils.hpp"
#include "librosa/util/exceptions.hpp"
#include "../internal/interp.hpp"
#include "../internal/fft.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <tuple>

namespace librosa {

// ============================================================================
// Window Functions
// ============================================================================

ArrayXr get_window(WindowType window_type, int n_fft, bool fftbins) {
    ArrayXr window(n_fft);
    int N = fftbins ? n_fft : n_fft - 1;

    switch (window_type) {
        case WindowType::Hann:
            for (int i = 0; i < n_fft; ++i) {
                window(i) = 0.5 - 0.5 * std::cos(2.0 * constants::PI * i / N);
            }
            break;

        case WindowType::Hamming:
            for (int i = 0; i < n_fft; ++i) {
                window(i) = 0.54 - 0.46 * std::cos(2.0 * constants::PI * i / N);
            }
            break;

        case WindowType::Blackman:
            for (int i = 0; i < n_fft; ++i) {
                window(i) = 0.42 - 0.5 * std::cos(2.0 * constants::PI * i / N)
                          + 0.08 * std::cos(4.0 * constants::PI * i / N);
            }
            break;

        case WindowType::Bartlett:
        case WindowType::Triangle:
            for (int i = 0; i < n_fft; ++i) {
                Real val = 2.0 * i / N;
                window(i) = (i <= N / 2) ? val : 2.0 - val;
            }
            break;

        case WindowType::Rectangular:
            window.setOnes();
            break;

        default:
            // Default to Hann
            for (int i = 0; i < n_fft; ++i) {
                window(i) = 0.5 - 0.5 * std::cos(2.0 * constants::PI * i / N);
            }
            break;
    }

    return window;
}

ArrayXr get_window(const std::string& window_name, int n_fft, bool fftbins) {
    if (window_name == "hann" || window_name == "hanning") {
        return get_window(WindowType::Hann, n_fft, fftbins);
    } else if (window_name == "hamming") {
        return get_window(WindowType::Hamming, n_fft, fftbins);
    } else if (window_name == "blackman") {
        return get_window(WindowType::Blackman, n_fft, fftbins);
    } else if (window_name == "bartlett") {
        return get_window(WindowType::Bartlett, n_fft, fftbins);
    } else if (window_name == "rectangular" || window_name == "boxcar" || window_name == "ones") {
        return get_window(WindowType::Rectangular, n_fft, fftbins);
    } else {
        throw ParameterError("Unknown window type: " + window_name);
    }
}

ArrayXr window_sumsquare(const ArrayXr& window, int n_frames, int hop_length, int n_fft) {
    int expected_length = n_fft + hop_length * (n_frames - 1);
    ArrayXr x = ArrayXr::Zero(expected_length);

    ArrayXr win_sq = window.square();

    for (int i = 0; i < n_frames; ++i) {
        int start = i * hop_length;
        x.segment(start, n_fft) += win_sq;
    }

    return x;
}

// ============================================================================
// STFT
// ============================================================================

ArrayXXc stft(const ArrayXr& y, int n_fft,
              std::optional<int> hop_length,
              std::optional<int> win_length,
              WindowType window,
              bool center,
              PadMode pad_mode) {
    util::valid_audio(y);

    int wl = win_length.value_or(n_fft);
    int hl = hop_length.value_or(wl / 4);

    if (hl < 1) {
        throw ParameterError("hop_length must be positive");
    }

    // Get window
    ArrayXr fft_window = get_window(window, wl, true);

    // Pad window to n_fft
    fft_window = util::pad_center(fft_window, n_fft);

    // Pad the signal if centering
    ArrayXr y_padded;
    if (center) {
        int pad_length = n_fft / 2;
        y_padded = util::pad_center(y, y.size() + 2 * pad_length, pad_mode);
    } else {
        if (n_fft > y.size()) {
            throw ParameterError("n_fft is too large for uncentered analysis");
        }
        y_padded = y;
    }

    // Frame the signal
    ArrayXXr y_frames = util::frame(y_padded, n_fft, hl);
    Eigen::Index n_frames = y_frames.cols();

    // Allocate output
    int n_freq = 1 + n_fft / 2;
    ArrayXXc stft_matrix(n_freq, n_frames);

    internal::RealFft fft(n_fft);
    std::vector<Real> fft_input(n_fft);
    std::vector<Complex> fft_output(n_freq);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        for (int i = 0; i < n_fft; ++i) {
            fft_input[i] = y_frames(i, t) * fft_window(i);
        }
        fft.forward(fft_input.data(), fft_output.data());
        for (int f = 0; f < n_freq; ++f) {
            stft_matrix(f, t) = fft_output[f];
        }
    }

    return stft_matrix;
}

ArrayXXc stft(const ArrayXr& y, int n_fft, int hop_length,
              const ArrayXr& window, bool center, PadMode pad_mode) {
    util::valid_audio(y);

    if (hop_length < 1) {
        throw ParameterError("hop_length must be positive");
    }

    // Window must already be of length n_fft
    ArrayXr fft_window = window;
    if (fft_window.size() != n_fft) {
        fft_window = util::pad_center(fft_window, n_fft);
    }

    // Pad the signal if centering
    ArrayXr y_padded;
    if (center) {
        int pad_length = n_fft / 2;
        y_padded = util::pad_center(y, y.size() + 2 * pad_length, pad_mode);
    } else {
        if (n_fft > y.size()) {
            throw ParameterError("n_fft is too large for uncentered analysis");
        }
        y_padded = y;
    }

    // Frame the signal
    ArrayXXr y_frames = util::frame(y_padded, n_fft, hop_length);
    Eigen::Index n_frames = y_frames.cols();

    // Allocate output
    int n_freq = 1 + n_fft / 2;
    ArrayXXc stft_matrix(n_freq, n_frames);

    internal::RealFft fft(n_fft);
    std::vector<Real> fft_input(n_fft);
    std::vector<Complex> fft_output(n_freq);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        for (int i = 0; i < n_fft; ++i) {
            fft_input[i] = y_frames(i, t) * fft_window(i);
        }
        fft.forward(fft_input.data(), fft_output.data());
        for (int f = 0; f < n_freq; ++f) {
            stft_matrix(f, t) = fft_output[f];
        }
    }

    return stft_matrix;
}

ArrayXr istft(const ArrayXXc& stft_matrix,
              std::optional<int> hop_length,
              std::optional<int> win_length,
              std::optional<int> n_fft_opt,
              WindowType window,
              bool center,
              std::optional<int> length) {

    Eigen::Index n_freq = stft_matrix.rows();
    Eigen::Index n_frames = stft_matrix.cols();

    int n_fft = n_fft_opt.value_or((n_freq - 1) * 2);
    int wl = win_length.value_or(n_fft);
    int hl = hop_length.value_or(wl / 4);

    // Get window
    ArrayXr ifft_window = get_window(window, wl, true);
    ifft_window = util::pad_center(ifft_window, n_fft);

    // Compute expected signal length
    int expected_length = n_fft + hl * (n_frames - 1);

    // Allocate output
    ArrayXr y = ArrayXr::Zero(expected_length);
    ArrayXr win_sum = ArrayXr::Zero(expected_length);

    internal::RealFft ifft(n_fft);
    std::vector<Real> ifft_output(n_fft);
    std::vector<Complex> ifft_input(n_freq);

    for (Eigen::Index t = 0; t < n_frames; ++t) {
        for (Eigen::Index f = 0; f < n_freq; ++f) {
            ifft_input[f] = stft_matrix(f, t);
        }
        ifft.inverse(ifft_input.data(), ifft_output.data());

        int start = t * hl;
        for (int i = 0; i < n_fft; ++i) {
            y(start + i) += ifft_window(i) * ifft_output[i] / n_fft;
            win_sum(start + i) += ifft_window(i) * ifft_window(i);
        }
    }

    // Normalize by window sum
    Real eps = util::tiny<Real>();
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (win_sum(i) > eps) {
            y(i) /= win_sum(i);
        }
    }

    // Trim if centering was used
    if (center) {
        int pad = n_fft / 2;
        y = y.segment(pad, y.size() - 2 * pad).eval();
    }

    // Trim or pad to desired length
    if (length) {
        y = util::fix_length(y, *length);
    }

    return y;
}

// ============================================================================
// Magnitude and Phase
// ============================================================================

std::pair<ArrayXXr, ArrayXXc> magphase(const ArrayXXc& D, Real power) {
    ArrayXXr mag = D.abs();

    // Compute phase as unit phasors
    ArrayXXc phase(D.rows(), D.cols());
    for (Eigen::Index i = 0; i < D.rows(); ++i) {
        for (Eigen::Index j = 0; j < D.cols(); ++j) {
            if (mag(i, j) == 0) {
                phase(i, j) = Complex(1.0, 0.0);
            } else {
                phase(i, j) = D(i, j) / mag(i, j);
            }
        }
    }

    // Apply power
    if (power != 1.0) {
        mag = mag.pow(power);
    }

    return {mag, phase};
}

ArrayXXr magnitude(const ArrayXXc& D) {
    return D.abs();
}

ArrayXXr phase(const ArrayXXc& D) {
    ArrayXXr angles(D.rows(), D.cols());
    for (Eigen::Index i = 0; i < D.rows(); ++i) {
        for (Eigen::Index j = 0; j < D.cols(); ++j) {
            angles(i, j) = std::arg(D(i, j));
        }
    }
    return angles;
}

// ============================================================================
// Decibel Conversions
// ============================================================================

ArrayXXr power_to_db(const ArrayXXr& S, Real ref, Real amin, std::optional<Real> top_db) {
    if (amin <= 0) {
        throw ParameterError("amin must be strictly positive");
    }

    Real ref_value = std::abs(ref);

    ArrayXXr log_spec = 10.0 * (S.max(amin).log10());
    log_spec -= 10.0 * std::log10(std::max(amin, ref_value));

    if (top_db) {
        if (*top_db < 0) {
            throw ParameterError("top_db must be non-negative");
        }
        Real max_val = log_spec.maxCoeff();
        log_spec = log_spec.max(max_val - *top_db);
    }

    return log_spec;
}

ArrayXXr power_to_db(const ArrayXXr& S,
                     std::function<Real(const ArrayXXr&)> ref,
                     Real amin, std::optional<Real> top_db) {
    Real ref_value = ref(S);
    return power_to_db(S, ref_value, amin, top_db);
}

Real power_to_db(Real S, Real ref, Real amin, std::optional<Real> top_db) {
    if (amin <= 0) {
        throw ParameterError("amin must be strictly positive");
    }

    Real ref_value = std::abs(ref);
    Real log_spec = 10.0 * std::log10(std::max(amin, std::abs(S)));
    log_spec -= 10.0 * std::log10(std::max(amin, ref_value));

    if (top_db) {
        if (*top_db < 0) {
            throw ParameterError("top_db must be non-negative");
        }
        log_spec = std::max(log_spec, -(*top_db));
    }

    return log_spec;
}

ArrayXXr db_to_power(const ArrayXXr& S_db, Real ref) {
    return ref * Eigen::pow(10.0, S_db * 0.1);
}

Real db_to_power(Real S_db, Real ref) {
    return ref * std::pow(10.0, S_db * 0.1);
}

ArrayXXr amplitude_to_db(const ArrayXXr& S, Real ref, Real amin, std::optional<Real> top_db) {
    return power_to_db(S.square(), ref * ref, amin * amin, top_db);
}

ArrayXXr amplitude_to_db(const ArrayXXr& S,
                         std::function<Real(const ArrayXXr&)> ref,
                         Real amin, std::optional<Real> top_db) {
    Real ref_value = ref(S);
    return amplitude_to_db(S, ref_value, amin, top_db);
}

Real amplitude_to_db(Real S, Real ref, Real amin, std::optional<Real> top_db) {
    return power_to_db(S * S, ref * ref, amin * amin, top_db);
}

ArrayXXr db_to_amplitude(const ArrayXXr& S_db, Real ref) {
    return (db_to_power(S_db, ref * ref)).sqrt();
}

Real db_to_amplitude(Real S_db, Real ref) {
    return std::sqrt(db_to_power(S_db, ref * ref));
}

// ============================================================================
// Perceptual Weighting
// ============================================================================

ArrayXXr perceptual_weighting(const ArrayXXr& S,
                              const ArrayXr& frequencies,
                              WeightType kind,
                              Real ref, Real amin,
                              std::optional<Real> top_db) {
    // Get frequency weights
    ArrayXr weights(frequencies.size());
    for (Eigen::Index i = 0; i < frequencies.size(); ++i) {
        switch (kind) {
            case WeightType::A:
                weights(i) = A_weighting(frequencies(i));
                break;
            case WeightType::B:
                weights(i) = B_weighting(frequencies(i));
                break;
            case WeightType::C:
                weights(i) = C_weighting(frequencies(i));
                break;
            case WeightType::D:
                weights(i) = D_weighting(frequencies(i));
                break;
            case WeightType::Z:
                weights(i) = 0.0;
                break;
        }
    }

    // Convert to dB
    ArrayXXr S_db = power_to_db(S, ref, amin, top_db);

    // Add weights
    for (Eigen::Index j = 0; j < S_db.cols(); ++j) {
        S_db.col(j) += weights;
    }

    return S_db;
}

// ============================================================================
// Phase Vocoder
// ============================================================================

ArrayXXc phase_vocoder(const ArrayXXc& D, Real rate,
                       std::optional<int> hop_length,
                       std::optional<int> n_fft_opt) {

    Eigen::Index n_freq = D.rows();
    Eigen::Index time_steps_orig = D.cols();

    int n_fft = n_fft_opt.value_or((n_freq - 1) * 2);
    int hl = hop_length.value_or(n_fft / 4);

    // Number of output time steps
    Eigen::Index time_steps = static_cast<Eigen::Index>(std::ceil(time_steps_orig / rate));

    // Phase accumulator
    ArrayXr phi_advance = ArrayXr::LinSpaced(n_freq, 0, constants::PI * hl);

    ArrayXXc d_stretch(n_freq, time_steps);

    // Initialize phase
    ArrayXr phase_acc(n_freq);
    for (Eigen::Index f = 0; f < n_freq; ++f) {
        phase_acc(f) = std::arg(D(f, 0));
    }

    for (Eigen::Index t = 0; t < time_steps; ++t) {
        Real t_interp = t * rate;
        Eigen::Index t0 = static_cast<Eigen::Index>(t_interp);
        Eigen::Index t1 = std::min(t0 + 1, time_steps_orig - 1);
        Real alpha = t_interp - t0;

        // Interpolate magnitudes
        ArrayXr mag0 = D.col(t0).abs();
        ArrayXr mag1 = D.col(t1).abs();
        ArrayXr mag = (1 - alpha) * mag0 + alpha * mag1;

        // Update phase
        for (Eigen::Index f = 0; f < n_freq; ++f) {
            d_stretch(f, t) = std::polar(mag(f), phase_acc(f));
        }

        if (t < time_steps - 1) {
            // Compute phase advance
            ArrayXr phase0(n_freq), phase1(n_freq);
            for (Eigen::Index f = 0; f < n_freq; ++f) {
                phase0(f) = std::arg(D(f, t0));
                phase1(f) = std::arg(D(f, t1));
            }

            // Phase difference
            ArrayXr dphi = phase1 - phase0 - phi_advance;

            // Wrap to [-pi, pi]
            for (Eigen::Index f = 0; f < n_freq; ++f) {
                dphi(f) = std::fmod(dphi(f) + constants::PI, constants::TWO_PI) - constants::PI;
            }

            // Update phase accumulator
            phase_acc += phi_advance + dphi;
        }
    }

    return d_stretch;
}

// ============================================================================
// Griffin-Lim Algorithm
// ============================================================================

ArrayXr griffinlim(const ArrayXXr& S, int n_iter,
                   std::optional<int> hop_length,
                   std::optional<int> win_length,
                   std::optional<int> n_fft,
                   WindowType window,
                   bool center,
                   std::optional<int> length,
                   PadMode pad_mode,
                   Real momentum,
                   const std::string& init_phase,
                   std::optional<unsigned int> random_state) {

    int nf = n_fft.value_or((S.rows() - 1) * 2);

    // Initialize with random phase
    ArrayXXc angles(S.rows(), S.cols());

    if (init_phase == "random") {
        std::mt19937 gen(random_state.value_or(std::random_device{}()));
        std::uniform_real_distribution<Real> dist(-constants::PI, constants::PI);

        for (Eigen::Index i = 0; i < S.rows(); ++i) {
            for (Eigen::Index j = 0; j < S.cols(); ++j) {
                angles(i, j) = std::polar(1.0, dist(gen));
            }
        }
    } else {
        angles.setOnes();
    }

    // Build complex spectrogram
    ArrayXXc rebuilt(S.rows(), S.cols());
    for (Eigen::Index i = 0; i < S.rows(); ++i) {
        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            rebuilt(i, j) = S(i, j) * angles(i, j);
        }
    }

    ArrayXXc tprev = rebuilt;

    for (int i = 0; i < n_iter; ++i) {
        // ISTFT
        ArrayXr y = istft(rebuilt, hop_length, win_length, nf, window, center, length);

        // STFT
        ArrayXXc inverse = stft(y, nf, hop_length, win_length, window, center, pad_mode);

        // Apply momentum
        ArrayXXc angles_new(S.rows(), S.cols());
        for (Eigen::Index r = 0; r < S.rows(); ++r) {
            for (Eigen::Index c = 0; c < S.cols(); ++c) {
                Real mag = std::abs(inverse(r, c));
                if (mag > util::tiny<Real>()) {
                    angles_new(r, c) = inverse(r, c) / mag;
                } else {
                    angles_new(r, c) = Complex(1.0, 0.0);
                }
            }
        }

        // Apply momentum
        if (momentum > 0 && i > 0) {
            for (Eigen::Index r = 0; r < S.rows(); ++r) {
                for (Eigen::Index c = 0; c < S.cols(); ++c) {
                    Real mag_t = std::abs(tprev(r, c));
                    Complex phase_t = (mag_t > util::tiny<Real>()) ?
                                      tprev(r, c) / mag_t : Complex(1.0, 0.0);
                    angles_new(r, c) = angles_new(r, c) - momentum * phase_t;
                    // Normalize
                    Real norm = std::abs(angles_new(r, c));
                    if (norm > util::tiny<Real>()) {
                        angles_new(r, c) /= norm;
                    }
                }
            }
        }

        // Store previous
        tprev = rebuilt;

        // Update rebuilt
        for (Eigen::Index r = 0; r < S.rows(); ++r) {
            for (Eigen::Index c = 0; c < S.cols(); ++c) {
                rebuilt(r, c) = S(r, c) * angles_new(r, c);
            }
        }
    }

    return istft(rebuilt, hop_length, win_length, nf, window, center, length);
}

// ============================================================================
// PCEN
// ============================================================================

ArrayXXr pcen(const ArrayXXr& S, Real sr, int hop_length,
              Real gain, Real bias, Real power,
              Real time_constant, Real eps) {

    // Compute smoothing coefficient using the formula from:
    // Lostanlen et al. "Per-Channel Energy Normalization: Why and How."
    // b = (sqrt(1 + 4*T^2) - 1) / (2*T^2)
    Real t_frames = time_constant * sr / static_cast<Real>(hop_length);
    Real b = (std::sqrt(1.0 + 4.0 * t_frames * t_frames) - 1.0) / (2.0 * t_frames * t_frames);

    // Compute lfilter_zi: steady-state initial condition for lfilter([b], [1, b-1])
    // For a first-order IIR filter y[t] = b*x[t] + (1-b)*y[t-1],
    // the steady-state response to a constant input is: zi = 1.0
    // (scipy.signal.lfilter_zi([b], [1, b-1]) returns [1.0])
    Real zi = 1.0;

    // Apply IIR filter: y[t] = b * x[t] + (1 - b) * y[t - 1]
    // with initial condition y[-1] = zi * x[0]
    ArrayXXr S_smooth(S.rows(), S.cols());
    for (Eigen::Index f = 0; f < S.rows(); ++f) {
        Real prev = zi;  // Initial condition: raw filter state (matching scipy lfilter_zi)
        for (Eigen::Index t = 0; t < S.cols(); ++t) {
            Real val = b * S(f, t) + (1.0 - b) * prev;
            S_smooth(f, t) = val;
            prev = val;
        }
    }

    // Adaptive gain control (working in log-space for stability, matching Python)
    // smooth = exp(-gain * (log(eps) + log1p(S_smooth / eps)))
    ArrayXXr smooth = Eigen::exp(-gain * (std::log(eps) + (S_smooth / eps + 1.0).log()));

    // Dynamic range compression
    // S_out = (bias^power) * expm1(power * log1p(S * smooth / bias))
    ArrayXXr result;
    if (power == 0.0) {
        result = (S * smooth + 1.0).log();
    } else if (bias == 0.0) {
        result = Eigen::exp(power * (S.log() + smooth.log()));
    } else {
        result = std::pow(bias, power) * (Eigen::exp(power * (S * smooth / bias + 1.0).log()) - 1.0);
    }

    return result;
}

// ============================================================================
// IIR Time-Frequency Representation
// ============================================================================

ArrayXXr iirt(const ArrayXr& y, Real sr, int win_length,
              std::optional<int> hop_length_opt, bool center,
              Real tuning, PadMode pad_mode) {
    util::valid_audio(y);

    int hl = hop_length_opt.value_or(win_length / 4);

    // Pad the signal if centering
    ArrayXr y_pad;
    if (center) {
        Eigen::Index pad_length = win_length / 2;
        y_pad.resize(y.size() + 2 * pad_length);
        // Pad according to mode
        switch (pad_mode) {
            case PadMode::Constant:
                y_pad.setZero();
                y_pad.segment(pad_length, y.size()) = y;
                break;
            default:
                y_pad = util::pad_center(y, y.size() + 2 * pad_length, pad_mode);
                break;
        }
    } else {
        y_pad = y;
    }

    // Get semitone filterbank
    auto [filterbank, sample_rates] = filters::semitone_filterbank(std::nullopt, tuning);
    int n_filters = static_cast<int>(filterbank.size());

    // Find unique sample rates
    std::vector<Real> unique_srs;
    for (Eigen::Index i = 0; i < sample_rates.size(); ++i) {
        bool found = false;
        for (Real usr : unique_srs) {
            if (std::abs(usr - sample_rates(i)) < 1e-6) {
                found = true;
                break;
            }
        }
        if (!found) unique_srs.push_back(sample_rates(i));
    }
    std::sort(unique_srs.begin(), unique_srs.end());

    // Create resampled versions
    std::unordered_map<int, ArrayXr> y_resampled;
    for (Real usr : unique_srs) {
        int key = static_cast<int>(std::round(usr));
        if (std::abs(usr - sr) < 1e-6) {
            y_resampled[key] = y_pad;
        } else {
            y_resampled[key] = resample(y_pad, sr, usr);
        }
    }

    // Compute number of frames
    int n_frames = 1 + static_cast<int>((y_pad.size() - win_length) / hl);
    if (n_frames < 1) n_frames = 1;

    // Pre-allocate output
    ArrayXXr bands_power(n_filters, n_frames);

    for (int i = 0; i < n_filters; ++i) {
        Real cur_sr = sample_rates(i);
        int sr_key = static_cast<int>(std::round(cur_sr));

        // Filter the resampled signal
        ArrayXr filtered = filters::sosfiltfilt(filterbank[i], y_resampled[sr_key]);

        Real factor = sr / cur_sr;
        Real hop_stmsp = static_cast<Real>(hl) / factor;
        int win_stmsp = static_cast<int>(std::round(static_cast<Real>(win_length) / factor));

        // Compute start indices using floating-point hop, then round
        std::vector<int> start_idx;
        for (int f = 0; f < n_frames; ++f) {
            int idx = static_cast<int>(std::round(f * hop_stmsp));
            start_idx.push_back(idx);
        }

        // Extend filtered signal if needed
        int min_length = (start_idx.empty() ? 0 : start_idx.back()) + win_stmsp;
        if (filtered.size() < min_length) {
            filtered = util::fix_length(filtered, min_length);
        }

        // Compute short-time mean-square power for each frame
        for (int f = 0; f < n_frames; ++f) {
            int start = start_idx[f];
            int end = std::min(start + win_stmsp, static_cast<int>(filtered.size()));
            Real power = 0.0;
            for (int j = start; j < end; ++j) {
                power += filtered(j) * filtered(j);
            }
            bands_power(i, f) = factor * power;
        }
    }

    return bands_power;
}

// ============================================================================
// Reassigned Spectrogram
// ============================================================================

std::tuple<ArrayXXr, ArrayXXr, ArrayXXr>
reassigned_spectrogram(const ArrayXr& y, Real sr, int n_fft,
                       std::optional<int> hop_length_opt,
                       std::optional<int> win_length_opt,
                       WindowType window, bool center,
                       bool reassign_frequencies, bool reassign_times,
                       Real ref_power, bool fill_nan, bool clip,
                       PadMode pad_mode) {

    if (ref_power < 0) {
        throw ParameterError("ref_power must be non-negative");
    }
    if (!reassign_frequencies && !reassign_times) {
        throw ParameterError("reassign_frequencies or reassign_times must be True");
    }

    int wl = win_length_opt.value_or(n_fft);
    int hl = hop_length_opt.value_or(wl / 4);

    // Get window and pad to n_fft
    ArrayXr win = get_window(window, wl, true);
    win = util::pad_center(win, n_fft);

    // Compute base STFT
    ArrayXXc S_h = stft(y, n_fft, hl, win, center, pad_mode);

    int n_freq = static_cast<int>(S_h.rows());
    Eigen::Index n_frames = S_h.cols();
    ArrayXXr mags = S_h.abs();

    // Initialize outputs
    ArrayXXr freqs(n_freq, n_frames);
    ArrayXXr times(n_freq, n_frames);

    // Frequency reassignment
    if (reassign_frequencies) {
        // Derivative window via cyclic_gradient
        ArrayXr win_deriv = util::cyclic_gradient(win, 1);

        ArrayXXc S_dh = stft(y, n_fft, hl, win_deriv, center, pad_mode);

        // Compute base frequencies
        ArrayXr bin_freqs = fft_frequencies(sr, n_fft);

        // correction = -imag(S_dh / S_h)
        // freqs = bin_freqs + correction * sr / (2*pi)
        for (Eigen::Index f = 0; f < n_freq; ++f) {
            for (Eigen::Index t = 0; t < n_frames; ++t) {
                if (std::abs(S_h(f, t)) > 0) {
                    Real correction = -(S_dh(f, t) / S_h(f, t)).imag();
                    freqs(f, t) = bin_freqs(f) + correction * sr / (2.0 * constants::PI);
                } else {
                    freqs(f, t) = std::numeric_limits<Real>::quiet_NaN();
                }
            }
        }
    }

    // Time reassignment
    if (reassign_times) {
        // Time-weighted window
        int half_width = n_fft / 2;
        ArrayXr window_times(n_fft);
        if (n_fft % 2 == 1) {
            for (int i = 0; i < n_fft; ++i) {
                window_times(i) = static_cast<Real>(i - half_width);
            }
        } else {
            for (int i = 0; i < n_fft; ++i) {
                window_times(i) = 0.5 - half_width + i;
            }
        }

        ArrayXr win_time_weighted = win * window_times;

        ArrayXXc S_th = stft(y, n_fft, hl, win_time_weighted, center, pad_mode);

        // Compute frame times
        ArrayXr frame_idx(n_frames);
        for (Eigen::Index t = 0; t < n_frames; ++t) {
            frame_idx(t) = static_cast<Real>(t);
        }

        std::optional<int> n_fft_pad;
        if (!center) {
            n_fft_pad = n_fft;
        }
        ArrayXr frame_times = frames_to_time(frame_idx, sr, hl, n_fft_pad);

        // correction = real(S_th / S_h)
        // times = frame_times + correction / sr
        for (Eigen::Index f = 0; f < n_freq; ++f) {
            for (Eigen::Index t = 0; t < n_frames; ++t) {
                if (std::abs(S_h(f, t)) > 0) {
                    Real correction = (S_th(f, t) / S_h(f, t)).real();
                    times(f, t) = frame_times(t) + correction / sr;
                } else {
                    times(f, t) = std::numeric_limits<Real>::quiet_NaN();
                }
            }
        }
    }

    // Threshold: find bins below ref_power
    Real ref_mag = std::sqrt(ref_power);

    if (reassign_frequencies) {
        if (ref_power > 0) {
            for (Eigen::Index f = 0; f < n_freq; ++f) {
                for (Eigen::Index t = 0; t < n_frames; ++t) {
                    if (!std::isnan(mags(f, t)) && mags(f, t) < ref_mag) {
                        freqs(f, t) = std::numeric_limits<Real>::quiet_NaN();
                    }
                }
            }
        }

        if (fill_nan) {
            ArrayXr bin_freqs = fft_frequencies(sr, n_fft);
            for (Eigen::Index f = 0; f < n_freq; ++f) {
                for (Eigen::Index t = 0; t < n_frames; ++t) {
                    if (std::isnan(freqs(f, t))) {
                        freqs(f, t) = bin_freqs(f);
                    }
                }
            }
        }

        if (clip) {
            freqs = freqs.max(0.0).min(sr / 2.0);
        }
    } else {
        // Return bin frequencies for all cells
        ArrayXr bin_freqs = fft_frequencies(sr, n_fft);
        for (Eigen::Index f = 0; f < n_freq; ++f) {
            freqs.row(f).setConstant(bin_freqs(f));
        }
    }

    if (reassign_times) {
        if (ref_power > 0) {
            for (Eigen::Index f = 0; f < n_freq; ++f) {
                for (Eigen::Index t = 0; t < n_frames; ++t) {
                    if (!std::isnan(mags(f, t)) && mags(f, t) < ref_mag) {
                        times(f, t) = std::numeric_limits<Real>::quiet_NaN();
                    }
                }
            }
        }

        if (fill_nan) {
            ArrayXr frame_idx(n_frames);
            for (Eigen::Index t = 0; t < n_frames; ++t) {
                frame_idx(t) = static_cast<Real>(t);
            }
            std::optional<int> n_fft_pad;
            if (!center) n_fft_pad = n_fft;
            ArrayXr frame_times = frames_to_time(frame_idx, sr, hl, n_fft_pad);

            for (Eigen::Index f = 0; f < n_freq; ++f) {
                for (Eigen::Index t = 0; t < n_frames; ++t) {
                    if (std::isnan(times(f, t))) {
                        times(f, t) = frame_times(t);
                    }
                }
            }
        }

        if (clip) {
            Real max_time = static_cast<Real>(y.size()) / sr;
            times = times.max(0.0).min(max_time);
        }
    } else {
        // Return frame times for all cells
        ArrayXr frame_idx(n_frames);
        for (Eigen::Index t = 0; t < n_frames; ++t) {
            frame_idx(t) = static_cast<Real>(t);
        }
        std::optional<int> n_fft_pad;
        if (!center) n_fft_pad = n_fft;
        ArrayXr frame_times = frames_to_time(frame_idx, sr, hl, n_fft_pad);
        for (Eigen::Index t = 0; t < n_frames; ++t) {
            times.col(t).setConstant(frame_times(t));
        }
    }

    return {freqs, times, mags};
}

// ============================================================================
// Fast Mellin Transform
// ============================================================================

ArrayXc fmt(const ArrayXr& y, Real t_min, std::optional<int> n_fmt_opt,
            const std::string& kind, Real beta, Real over_sample) {

    Eigen::Index n = y.size();

    if (n < 3) {
        throw ParameterError("y must have at least 3 samples");
    }
    if (t_min <= 0) {
        throw ParameterError("t_min must be positive");
    }
    if (!y.isFinite().all()) {
        throw ParameterError("y must be finite everywhere");
    }

    // Compute log_base and n_fmt
    Real log_base;
    int n_fmt;

    if (!n_fmt_opt.has_value()) {
        if (over_sample < 1.0) {
            throw ParameterError("over_sample must be >= 1");
        }
        log_base = std::log(static_cast<Real>(n - 1)) - std::log(static_cast<Real>(n - 2));
        n_fmt = static_cast<int>(std::ceil(
            over_sample * (std::log(static_cast<Real>(n - 1)) - std::log(t_min)) / log_base));
    } else {
        n_fmt = n_fmt_opt.value();
        if (n_fmt < 3) {
            throw ParameterError("n_fmt must be >= 3");
        }
        log_base = (std::log(static_cast<Real>(n_fmt - 1)) - std::log(static_cast<Real>(n_fmt - 2))) / over_sample;
    }

    Real base = std::exp(log_base);

    // Original grid: [0, 1/n, 2/n, ..., (n-1)/n]
    ArrayXr x = ArrayXr::LinSpaced(n, 0, static_cast<Real>(n - 1) / n);

    // Build interpolator
    internal::Interp1d::Kind interp_kind = internal::Interp1d::Kind::Cubic;
    if (kind == "linear") {
        interp_kind = internal::Interp1d::Kind::Linear;
    }
    internal::Interp1d f_interp(x, y, interp_kind);

    // Build exponential sampling grid
    int n_over = static_cast<int>(std::ceil(over_sample));
    int total_pts = n_fmt + n_over;
    Real log_start = (std::log(t_min) - std::log(static_cast<Real>(n))) / log_base;
    Real log_end = 0.0;

    ArrayXr x_exp(n_fmt);
    for (int i = 0; i < n_fmt; ++i) {
        Real exponent = log_start + (log_end - log_start) * static_cast<Real>(i) / total_pts;
        x_exp(i) = std::pow(base, exponent);
    }

    // Clip to interpolation range
    Real x_min = t_min / n;
    Real x_max = x(n - 1);
    for (Eigen::Index i = 0; i < n_fmt; ++i) {
        x_exp(i) = std::max(x_min, std::min(x_max, x_exp(i)));
    }

    // Resample
    ArrayXr y_res = f_interp(x_exp);

    // Apply window and normalization
    ArrayXr windowed(n_fmt);
    Real sqrt_n = std::sqrt(static_cast<Real>(n));
    for (int i = 0; i < n_fmt; ++i) {
        windowed(i) = y_res(i) * std::pow(x_exp(i), beta) * sqrt_n / n_fmt;
    }

    int n_freq = n_fmt / 2 + 1;
    std::vector<Real> fft_input(n_fmt);
    for (int i = 0; i < n_fmt; ++i) {
        fft_input[i] = windowed(i);
    }

    ArrayXc result(n_freq);
    internal::RealFft fft(n_fmt);
    fft.forward(fft_input.data(), result.data());

    return result;
}

} // namespace librosa
