#include <librosa/feature/rhythm.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/harmonic.hpp>
#include <librosa/onset.hpp>
#include <librosa/beat.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include <algorithm>
#include <cmath>

namespace librosa {
namespace feature {

ArrayXXc fourier_tempogram(const ArrayXr& onset_envelope,
                           Real sr, int hop_length,
                           int win_length, bool center,
                           WindowType window) {
    if (win_length < 1) {
        throw ParameterError("win_length must be a positive integer");
    }

    // The Fourier tempogram is just the STFT of the onset envelope
    // with hop_length=1 (frame-by-frame)
    return stft(onset_envelope, win_length, 1 /* hop_length for stft */,
                std::nullopt, window, center);
}

ArrayXXc fourier_tempogram_audio(const ArrayXr& y,
                                 Real sr, int hop_length,
                                 int win_length, bool center,
                                 WindowType window) {
    // Compute onset envelope
    ArrayXr envelope = onset::onset_strength(y, sr, 2048, hop_length);

    return fourier_tempogram(envelope, sr, hop_length, win_length, center, window);
}

ArrayXXr tempogram_ratio(const ArrayXXr& tg, Real sr,
                         int hop_length,
                         const std::vector<Real>& factors_in,
                         Real start_bpm, Real std_bpm,
                         std::optional<Real> max_tempo,
                         Real fill_value) {
    // Default factors: common musical tempo ratios
    std::vector<Real> factors = factors_in;
    if (factors.empty()) {
        factors = {4.0, 8.0/3, 3.0, 2.0, 4.0/3, 3.0/2, 1.0,
                   2.0/3, 3.0/4, 1.0/2, 1.0/3, 3.0/8, 1.0/4};
    }

    Eigen::Index win_length = tg.rows();
    Eigen::Index n_frames = tg.cols();

    // Get BPM frequencies for the tempogram bins
    ArrayXr bpms = tempo_frequencies(static_cast<int>(win_length), hop_length, sr);

    // Estimate per-frame tempo
    ArrayXr tempos(n_frames);

    // Compute log-normal prior
    ArrayXr logprior(win_length);
    for (Eigen::Index i = 0; i < win_length; ++i) {
        if (bpms(i) > 0) {
            Real log_ratio = (std::log2(bpms(i)) - std::log2(start_bpm)) / std_bpm;
            logprior(i) = -0.5 * log_ratio * log_ratio;
        } else {
            logprior(i) = -std::numeric_limits<Real>::infinity();
        }
    }

    if (max_tempo.has_value()) {
        for (Eigen::Index i = 0; i < win_length; ++i) {
            if (bpms(i) > max_tempo.value()) {
                logprior(i) = -std::numeric_limits<Real>::infinity();
            }
        }
    }

    // Per-frame tempo estimation
    for (Eigen::Index f = 0; f < n_frames; ++f) {
        ArrayXr weighted(win_length);
        for (Eigen::Index i = 0; i < win_length; ++i) {
            weighted(i) = std::log1p(1e6 * tg(i, f)) + logprior(i);
        }
        Eigen::Index best_idx;
        weighted.maxCoeff(&best_idx);
        tempos(f) = bpms(best_idx);
    }

    // Use f0_harmonics to extract energy at harmonic factors of estimated tempo
    ArrayXXr result = core::f0_harmonics(tg, tempos, bpms, factors, fill_value);

    return result;
}

} // namespace feature
} // namespace librosa
