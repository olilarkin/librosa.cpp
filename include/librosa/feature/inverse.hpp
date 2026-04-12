#pragma once

#include "../types.hpp"
#include <optional>

namespace librosa {
namespace feature {

/// Approximate an STFT magnitude from a mel power spectrogram using NNLS
///
/// Inverts the mel filterbank mapping to recover linear-frequency magnitudes.
///
/// @param M Mel power spectrogram [shape: (n_mels, t)]
/// @param sr Sample rate
/// @param n_fft FFT window size
/// @param power Exponent for the mel spectrogram (2 for power, 1 for energy)
/// @return Approximate STFT magnitude [shape: (n_fft/2+1, t)]
ArrayXXr mel_to_stft(const ArrayXXr& M, Real sr = 22050, int n_fft = 2048,
                     Real power = 2.0);

/// Invert a mel power spectrogram to audio using Griffin-Lim
///
/// @param M Mel power spectrogram [shape: (n_mels, t)]
/// @param sr Sample rate
/// @param n_fft FFT window size
/// @param hop_length Samples between frames
/// @param win_length Window length
/// @param window Window type
/// @param center Center frames
/// @param power Exponent for the mel spectrogram
/// @param n_iter Number of Griffin-Lim iterations
/// @param length Desired output length
/// @return Reconstructed audio time series
ArrayXr mel_to_audio(const ArrayXXr& M, Real sr = 22050, int n_fft = 2048,
                     std::optional<int> hop_length = std::nullopt,
                     std::optional<int> win_length = std::nullopt,
                     WindowType window = WindowType::Hann, bool center = true,
                     Real power = 2.0, int n_iter = 32,
                     std::optional<int> length = std::nullopt);

/// Invert MFCCs to an approximate mel power spectrogram
///
/// Applies inverse liftering (if any), inverse DCT, and dB-to-power conversion.
///
/// @param mfcc MFCC matrix [shape: (n_mfcc, t)]
/// @param n_mels Number of mel bands to produce
/// @param dct_type DCT type used in forward MFCC (2 or 3)
/// @param ortho_norm Whether orthonormal DCT normalization was used
/// @param ref Reference power for dB conversion
/// @param lifter Liftering coefficient used in forward MFCC
/// @return Approximate mel power spectrogram [shape: (n_mels, t)]
ArrayXXr mfcc_to_mel(const ArrayXXr& mfcc, int n_mels = 128,
                     int dct_type = 2, bool ortho_norm = true,
                     Real ref = 1.0, Real lifter = 0);

/// Invert MFCCs to audio via mel → STFT → Griffin-Lim
///
/// @param mfcc MFCC matrix [shape: (n_mfcc, t)]
/// @param n_mels Number of mel bands
/// @param dct_type DCT type used in forward MFCC
/// @param ortho_norm Whether orthonormal normalization was used
/// @param ref Reference power for dB conversion
/// @param lifter Liftering coefficient
/// @param sr Sample rate
/// @param n_fft FFT window size
/// @param hop_length Samples between frames
/// @param n_iter Number of Griffin-Lim iterations
/// @return Reconstructed audio time series
ArrayXr mfcc_to_audio(const ArrayXXr& mfcc, int n_mels = 128,
                      int dct_type = 2, bool ortho_norm = true,
                      Real ref = 1.0, Real lifter = 0,
                      Real sr = 22050, int n_fft = 2048,
                      std::optional<int> hop_length = std::nullopt,
                      int n_iter = 32);

} // namespace feature
} // namespace librosa
