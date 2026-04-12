#pragma once

#include "../types.hpp"
#include <optional>

namespace librosa {
namespace feature {

// ============================================================================
// Mel Spectrogram
// ============================================================================

/// Compute a mel-scaled spectrogram
/// @param y Audio time series
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param win_length Window length (default: n_fft)
/// @param window Window type
/// @param center If true, center frames
/// @param pad_mode Padding mode
/// @param power Exponent for magnitude (1 for energy, 2 for power)
/// @param n_mels Number of mel bands
/// @param fmin Minimum frequency
/// @param fmax Maximum frequency (default: sr/2)
/// @param htk Use HTK formula
/// @param norm Normalization mode for mel filterbank ("slaney" or none)
/// @return Mel spectrogram [shape: (n_mels, t)]
ArrayXXr melspectrogram(
    const ArrayXr& y,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    std::optional<int> win_length = std::nullopt,
    WindowType window = WindowType::Hann,
    bool center = true,
    PadMode pad_mode = PadMode::Constant,
    Real power = 2.0,
    int n_mels = 128,
    Real fmin = 0.0,
    std::optional<Real> fmax = std::nullopt,
    bool htk = false,
    bool norm_slaney = true);

/// Compute a mel-scaled spectrogram from pre-computed spectrogram
/// @param S Spectrogram magnitude
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param n_mels Number of mel bands
/// @param fmin Minimum frequency
/// @param fmax Maximum frequency (default: sr/2)
/// @param htk Use HTK formula
/// @param norm_slaney Use Slaney normalization
/// @return Mel spectrogram [shape: (n_mels, t)]
ArrayXXr melspectrogram(
    const ArrayXXr& S,
    Real sr = 22050,
    int n_fft = 2048,
    int n_mels = 128,
    Real fmin = 0.0,
    std::optional<Real> fmax = std::nullopt,
    bool htk = false,
    bool norm_slaney = true);

// ============================================================================
// MFCC
// ============================================================================

/// Compute Mel-frequency cepstral coefficients
/// @param y Audio time series
/// @param sr Sampling rate
/// @param n_mfcc Number of MFCCs to return
/// @param dct_type DCT type (2 or 3)
/// @param norm_ortho Use orthonormal DCT basis
/// @param lifter Liftering parameter (0 = no liftering)
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param n_mels Number of mel bands
/// @param fmin Minimum frequency
/// @param fmax Maximum frequency
/// @param htk Use HTK formula
/// @return MFCC sequence [shape: (n_mfcc, t)]
ArrayXXr mfcc(
    const ArrayXr& y,
    Real sr = 22050,
    int n_mfcc = 20,
    int dct_type = 2,
    bool norm_ortho = true,
    Real lifter = 0,
    int n_fft = 2048,
    int hop_length = 512,
    int n_mels = 128,
    Real fmin = 0.0,
    std::optional<Real> fmax = std::nullopt,
    bool htk = false);

/// Compute MFCC from pre-computed log-power mel spectrogram
/// @param S Log-power Mel spectrogram
/// @param n_mfcc Number of MFCCs to return
/// @param dct_type DCT type (2 or 3)
/// @param norm_ortho Use orthonormal DCT basis
/// @param lifter Liftering parameter
/// @return MFCC sequence [shape: (n_mfcc, t)]
ArrayXXr mfcc(
    const ArrayXXr& S,
    int n_mfcc = 20,
    int dct_type = 2,
    bool norm_ortho = true,
    Real lifter = 0);

// ============================================================================
// Chroma
// ============================================================================

/// Compute chromagram from STFT
/// @param y Audio time series
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param n_chroma Number of chroma bins
/// @param tuning Deviation from A440 tuning (optional, will be estimated if not provided)
/// @param norm Normalization factor (inf for max norm)
/// @param window Window type
/// @param center If true, center frames
/// @return Chromagram [shape: (n_chroma, t)]
ArrayXXr chroma_stft(
    const ArrayXr& y,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    int n_chroma = 12,
    std::optional<Real> tuning = std::nullopt,
    Real norm = std::numeric_limits<Real>::infinity(),
    WindowType window = WindowType::Hann,
    bool center = true);

/// Compute chromagram from pre-computed spectrogram
/// @param S Power spectrogram
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param n_chroma Number of chroma bins
/// @param tuning Deviation from A440 tuning
/// @param norm Normalization factor
/// @return Chromagram [shape: (n_chroma, t)]
ArrayXXr chroma_stft(
    const ArrayXXr& S,
    Real sr = 22050,
    int n_fft = 2048,
    int n_chroma = 12,
    std::optional<Real> tuning = std::nullopt,
    Real norm = std::numeric_limits<Real>::infinity());

// ============================================================================
// Chroma CQT
// ============================================================================

/// Compute chromagram from constant-Q transform (from audio)
/// @param y Audio time series
/// @param sr Sampling rate
/// @param hop_length Number of samples between frames
/// @param fmin Minimum frequency (default: C1)
/// @param norm Normalization factor (inf for max norm, negative for no normalization)
/// @param threshold Minimum threshold for chroma values
/// @param tuning Deviation from A440 tuning (optional, will be estimated if not provided)
/// @param n_chroma Number of chroma bins
/// @param n_octaves Number of octaves to analyze
/// @param bins_per_octave Number of bins per octave in the CQT
/// @return Chromagram [shape: (n_chroma, t)]
ArrayXXr chroma_cqt(
    const ArrayXr& y,
    Real sr = 22050,
    int hop_length = 512,
    std::optional<Real> fmin = std::nullopt,
    Real norm = std::numeric_limits<Real>::infinity(),
    Real threshold = 0.0,
    std::optional<Real> tuning = std::nullopt,
    int n_chroma = 12,
    int n_octaves = 7,
    int bins_per_octave = 36);

/// Compute chromagram from pre-computed CQT magnitude
/// @param C CQT magnitude matrix
/// @param fmin Minimum frequency (default: C1)
/// @param norm Normalization factor (inf for max norm, negative for no normalization)
/// @param threshold Minimum threshold for chroma values
/// @param n_chroma Number of chroma bins
/// @param bins_per_octave Number of bins per octave
/// @return Chromagram [shape: (n_chroma, t)]
ArrayXXr chroma_cqt(
    const ArrayXXr& C,
    std::optional<Real> fmin = std::nullopt,
    Real norm = std::numeric_limits<Real>::infinity(),
    Real threshold = 0.0,
    int n_chroma = 12,
    int bins_per_octave = 36);

// ============================================================================
// Chroma CENS
// ============================================================================

/// Compute Chroma Energy Normalized Statistics (CENS) chromagram
/// Includes L1 normalization, quantization, temporal smoothing, and L2 normalization.
/// @param y Audio time series
/// @param sr Sampling rate
/// @param hop_length Number of samples between frames
/// @param fmin Minimum frequency (default: C1)
/// @param tuning Deviation from A440 tuning
/// @param n_chroma Number of chroma bins
/// @param n_octaves Number of octaves
/// @param bins_per_octave Bins per octave for CQT
/// @param norm Final normalization (default: L2)
/// @param win_len_smooth Smoothing window length (0 to disable)
/// @return CENS chromagram [shape: (n_chroma, t)]
ArrayXXr chroma_cens(
    const ArrayXr& y,
    Real sr = 22050,
    int hop_length = 512,
    std::optional<Real> fmin = std::nullopt,
    std::optional<Real> tuning = std::nullopt,
    int n_chroma = 12,
    int n_octaves = 7,
    int bins_per_octave = 36,
    Real norm = 2.0,
    int win_len_smooth = 41);

// ============================================================================
// Chroma VQT
// ============================================================================

/// Compute chromagram from variable-Q transform (from audio)
/// @param y Audio time series
/// @param sr Sampling rate
/// @param hop_length Number of samples between frames
/// @param fmin Minimum frequency (default: C1)
/// @param norm Normalization factor (inf for max norm)
/// @param threshold Minimum threshold for chroma values
/// @param n_octaves Number of octaves
/// @param bins_per_octave Bins per octave for VQT (also determines n_chroma output)
/// @param gamma Bandwidth offset for VQT
/// @return Chromagram [shape: (bins_per_octave, t)]
ArrayXXr chroma_vqt(
    const ArrayXr& y,
    Real sr = 22050,
    int hop_length = 512,
    std::optional<Real> fmin = std::nullopt,
    Real norm = std::numeric_limits<Real>::infinity(),
    Real threshold = 0.0,
    int n_octaves = 7,
    int bins_per_octave = 12,
    Real gamma = 0.0);

/// Compute chromagram from pre-computed VQT magnitude
/// @param V VQT magnitude matrix
/// @param fmin Minimum frequency (default: C1)
/// @param norm Normalization factor
/// @param threshold Minimum threshold
/// @param bins_per_octave Bins per octave (determines n_chroma output)
/// @return Chromagram [shape: (bins_per_octave, t)]
ArrayXXr chroma_vqt(
    const ArrayXXr& V,
    std::optional<Real> fmin = std::nullopt,
    Real norm = std::numeric_limits<Real>::infinity(),
    Real threshold = 0.0,
    int bins_per_octave = 12);

// ============================================================================
// Spectral Features
// ============================================================================

/// Compute spectral centroid
/// @param y Audio time series
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param window Window type
/// @param center If true, center frames
/// @return Spectral centroid [shape: (1, t)]
ArrayXXr spectral_centroid(
    const ArrayXr& y,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    WindowType window = WindowType::Hann,
    bool center = true);

/// Compute spectral centroid from pre-computed spectrogram
/// @param S Spectrogram magnitude
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param freq Optional custom frequency bins
/// @return Spectral centroid [shape: (1, t)]
ArrayXXr spectral_centroid(
    const ArrayXXr& S,
    Real sr = 22050,
    int n_fft = 2048,
    const ArrayXr* freq = nullptr);

/// Compute spectral bandwidth
/// @param y Audio time series
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param window Window type
/// @param center If true, center frames
/// @param p Power for bandwidth computation
/// @param norm Normalize per-frame energy
/// @return Spectral bandwidth [shape: (1, t)]
ArrayXXr spectral_bandwidth(
    const ArrayXr& y,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    WindowType window = WindowType::Hann,
    bool center = true,
    Real p = 2,
    bool norm = true);

/// Compute spectral bandwidth from pre-computed spectrogram
/// @param S Spectrogram magnitude
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param centroid Pre-computed centroid (optional)
/// @param freq Optional custom frequency bins
/// @param p Power for bandwidth computation
/// @param norm Normalize per-frame energy
/// @return Spectral bandwidth [shape: (1, t)]
ArrayXXr spectral_bandwidth(
    const ArrayXXr& S,
    Real sr = 22050,
    int n_fft = 2048,
    const ArrayXXr* centroid = nullptr,
    const ArrayXr* freq = nullptr,
    Real p = 2,
    bool norm = true);

/// Compute spectral rolloff frequency
/// @param y Audio time series
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param window Window type
/// @param center If true, center frames
/// @param roll_percent Roll-off percentage (0-1)
/// @return Roll-off frequency [shape: (1, t)]
ArrayXXr spectral_rolloff(
    const ArrayXr& y,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    WindowType window = WindowType::Hann,
    bool center = true,
    Real roll_percent = 0.85);

/// Compute spectral rolloff from pre-computed spectrogram
/// @param S Spectrogram magnitude
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param freq Optional custom frequency bins
/// @param roll_percent Roll-off percentage (0-1)
/// @return Roll-off frequency [shape: (1, t)]
ArrayXXr spectral_rolloff(
    const ArrayXXr& S,
    Real sr = 22050,
    int n_fft = 2048,
    const ArrayXr* freq = nullptr,
    Real roll_percent = 0.85);

/// Compute spectral flatness
/// @param y Audio time series
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param window Window type
/// @param center If true, center frames
/// @param amin Minimum threshold for numerical stability
/// @param power Exponent for magnitude
/// @return Spectral flatness [shape: (1, t)]
ArrayXXr spectral_flatness(
    const ArrayXr& y,
    int n_fft = 2048,
    int hop_length = 512,
    WindowType window = WindowType::Hann,
    bool center = true,
    Real amin = 1e-10,
    Real power = 2.0);

/// Compute spectral flatness from pre-computed spectrogram
/// @param S Spectrogram magnitude
/// @param amin Minimum threshold for numerical stability
/// @param power Exponent for magnitude
/// @return Spectral flatness [shape: (1, t)]
ArrayXXr spectral_flatness(
    const ArrayXXr& S,
    Real amin = 1e-10,
    Real power = 2.0);

/// Compute spectral contrast
/// @param y Audio time series
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param window Window type
/// @param center If true, center frames
/// @param fmin Frequency cutoff for first band
/// @param n_bands Number of frequency bands
/// @param quantile Quantile for peak/valley computation
/// @param linear Return linear difference instead of log
/// @return Spectral contrast [shape: (n_bands+1, t)]
ArrayXXr spectral_contrast(
    const ArrayXr& y,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    WindowType window = WindowType::Hann,
    bool center = true,
    Real fmin = 200.0,
    int n_bands = 6,
    Real quantile = 0.02,
    bool linear = false);

/// Compute spectral contrast from pre-computed spectrogram
/// @param S Spectrogram magnitude
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param freq Optional custom frequency bins
/// @param fmin Frequency cutoff for first band
/// @param n_bands Number of frequency bands
/// @param quantile Quantile for peak/valley computation
/// @param linear Return linear difference instead of log
/// @return Spectral contrast [shape: (n_bands+1, t)]
ArrayXXr spectral_contrast(
    const ArrayXXr& S,
    Real sr = 22050,
    int n_fft = 2048,
    const ArrayXr* freq = nullptr,
    Real fmin = 200.0,
    int n_bands = 6,
    Real quantile = 0.02,
    bool linear = false);

// ============================================================================
// RMS and Zero Crossing Rate
// ============================================================================

/// Compute root-mean-square energy
/// @param y Audio time series
/// @param frame_length Analysis frame length
/// @param hop_length Number of samples between frames
/// @param center If true, center frames
/// @return RMS energy [shape: (1, t)]
ArrayXXr rms(
    const ArrayXr& y,
    int frame_length = 2048,
    int hop_length = 512,
    bool center = true);

/// Compute RMS from pre-computed spectrogram
/// @param S Spectrogram magnitude
/// @param frame_length Analysis frame length
/// @return RMS energy [shape: (1, t)]
ArrayXXr rms(
    const ArrayXXr& S,
    int frame_length = 2048);

/// Compute zero crossing rate
/// @param y Audio time series
/// @param frame_length Analysis frame length
/// @param hop_length Number of samples between frames
/// @param center If true, center frames
/// @param threshold Threshold for zero crossing detection
/// @return Zero crossing rate [shape: (1, t)]
ArrayXXr zero_crossing_rate(
    const ArrayXr& y,
    int frame_length = 2048,
    int hop_length = 512,
    bool center = true,
    Real threshold = 0.0);

// ============================================================================
// Polynomial Features
// ============================================================================

/// Compute polynomial features from spectrogram
/// @param S Spectrogram magnitude
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param order Polynomial order
/// @param freq Optional custom frequency bins
/// @return Polynomial coefficients [shape: (order+1, t)]
ArrayXXr poly_features(
    const ArrayXXr& S,
    Real sr = 22050,
    int n_fft = 2048,
    int order = 1,
    const ArrayXr* freq = nullptr);

/// Compute polynomial features from audio
/// @param y Audio time series
/// @param sr Sampling rate
/// @param n_fft FFT window size
/// @param hop_length Number of samples between frames
/// @param order Polynomial order
/// @return Polynomial coefficients [shape: (order+1, t)]
ArrayXXr poly_features(
    const ArrayXr& y,
    Real sr = 22050,
    int n_fft = 2048,
    int hop_length = 512,
    int order = 1);

// ============================================================================
// Tonnetz
// ============================================================================

/// Compute tonal centroid features (tonnetz)
/// @param y Audio time series
/// @param sr Sampling rate
/// @param chroma Pre-computed chromagram (optional)
/// @return Tonnetz features [shape: (6, t)]
ArrayXXr tonnetz(
    const ArrayXr& y,
    Real sr = 22050,
    const ArrayXXr* chroma = nullptr);

/// Compute tonnetz from pre-computed chromagram
/// @param chroma Chromagram
/// @return Tonnetz features [shape: (6, t)]
ArrayXXr tonnetz(const ArrayXXr& chroma);

} // namespace feature
} // namespace librosa
