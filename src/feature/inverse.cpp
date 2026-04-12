#include <librosa/feature/inverse.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/filters.hpp>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include "../internal/dct.hpp"
#include <cmath>

namespace librosa {
namespace feature {

ArrayXXr mel_to_stft(const ArrayXXr& M, Real sr, int n_fft, Real power) {
    if (power <= 0) {
        throw ParameterError("power must be positive");
    }

    Eigen::Index n_mels = M.rows();

    // Build mel filter bank
    ArrayXXr mel_basis = filters::mel(sr, n_fft, static_cast<int>(n_mels));

    // Solve NNLS: find X >= 0 such that mel_basis * X ≈ M
    // mel_basis: (n_mels, n_fft/2+1), M: (n_mels, t)
    // We want X: (n_fft/2+1, t)
    ArrayXXr X = util::nnls(mel_basis.matrix(), M);

    // Apply inverse power: if M was in power domain, take root
    if (power != 1.0) {
        X = X.pow(1.0 / power);
    }

    return X;
}

ArrayXr mel_to_audio(const ArrayXXr& M, Real sr, int n_fft,
                     std::optional<int> hop_length,
                     std::optional<int> win_length,
                     WindowType window, bool center,
                     Real power, int n_iter,
                     std::optional<int> length) {
    // Recover STFT magnitude from mel
    ArrayXXr S = mel_to_stft(M, sr, n_fft, power);

    // Use Griffin-Lim to reconstruct audio from magnitude
    return griffinlim(S, n_iter, hop_length, win_length, n_fft, window,
                      center, length);
}

ArrayXXr mfcc_to_mel(const ArrayXXr& mfcc, int n_mels,
                     int dct_type, bool ortho_norm,
                     Real ref, Real lifter) {
    Eigen::Index n_mfcc = mfcc.rows();

    ArrayXXr M = mfcc;

    // Undo liftering if applied
    if (lifter > 0) {
        ArrayXr LI(n_mfcc);
        for (Eigen::Index n = 0; n < n_mfcc; ++n) {
            LI(n) = std::sin(M_PI * (n + 1) / lifter);
        }

        for (Eigen::Index t = 0; t < M.cols(); ++t) {
            for (Eigen::Index n = 0; n < n_mfcc; ++n) {
                Real scale = 1.0 + (lifter / 2.0) * LI(n);
                if (std::abs(scale) > 1e-10) {
                    M(n, t) /= scale;
                }
            }
        }
    }

    // Apply inverse DCT to go from cepstral domain back to log-mel domain
    // Forward MFCC with dct_type=2 → inverse uses dct_type=3 (and vice versa)
    ArrayXXr S_db;
    if (dct_type == 2) {
        S_db = internal::dct_iii(M, n_mels, ortho_norm);
    } else if (dct_type == 3) {
        S_db = internal::dct_ii(M, n_mels, ortho_norm);
    } else {
        throw ParameterError("dct_type must be 2 or 3");
    }

    // Convert from dB to power
    ArrayXXr mel_power = db_to_power(S_db, ref);

    return mel_power;
}

ArrayXr mfcc_to_audio(const ArrayXXr& mfcc, int n_mels,
                      int dct_type, bool ortho_norm,
                      Real ref, Real lifter,
                      Real sr, int n_fft,
                      std::optional<int> hop_length,
                      int n_iter) {
    // MFCC → mel power spectrogram
    ArrayXXr mel = mfcc_to_mel(mfcc, n_mels, dct_type, ortho_norm, ref, lifter);

    // mel → audio via Griffin-Lim
    return mel_to_audio(mel, sr, n_fft, hop_length, std::nullopt,
                        WindowType::Hann, true, 2.0, n_iter);
}

} // namespace feature
} // namespace librosa
