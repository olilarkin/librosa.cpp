#pragma once

#include "../types.hpp"
#include <optional>
#include <string>

namespace librosa {

// ============================================================================
// Audio I/O
// ============================================================================

struct AudioFileInfo {
    Eigen::Index samples;
    Real sample_rate;
    int channels;
    Real duration;
};

/// Load an audio file as floating point time series
/// @param path Path to audio file
/// @param sr Target sample rate (std::nullopt for native)
/// @param mono Convert to mono
/// @param offset Start reading after this time (seconds)
/// @param duration Only load this much audio (seconds)
/// @return AudioData structure with samples and metadata
AudioData load(const std::string& path,
               std::optional<Real> sr = 22050,
               bool mono = true,
               Real offset = 0.0,
               std::optional<Real> duration = std::nullopt);

/// Get duration of an audio file without loading
/// @param path Path to audio file
/// @return Duration in seconds
Real get_duration(const std::string& path);

/// Get audio file metadata without loading or transforming samples
/// @param path Path to audio file
/// @return Native file metadata
AudioFileInfo get_audio_info(const std::string& path);

/// Get duration from audio buffer
/// @param y Audio samples
/// @param sr Sample rate
/// @return Duration in seconds
Real get_duration(const ArrayXr& y, Real sr = 22050);

/// Get duration from spectrogram
/// @param S Spectrogram or feature matrix
/// @param sr Sample rate
/// @param hop_length Hop length
/// @param n_fft FFT size
/// @param center Whether STFT was centered
/// @return Duration in seconds
Real get_duration(const ArrayXXr& S, Real sr = 22050, int hop_length = 512,
                  int n_fft = 2048, bool center = true);

/// Get sample rate of an audio file without loading
/// @param path Path to audio file
/// @return Sample rate in Hz
Real get_samplerate(const std::string& path);

// ============================================================================
// Audio Processing
// ============================================================================

/// Convert multi-channel audio to mono by averaging
/// @param y Multi-channel audio (channels x samples)
/// @return Mono audio (samples)
ArrayXr to_mono(const ArrayXXr& y);
ArrayXr to_mono(const ArrayXr& y);  // Pass-through for already mono

/// Resample a time series
/// @param y Input signal
/// @param orig_sr Original sample rate
/// @param target_sr Target sample rate
/// @param res_type Resampling method ("kaiser_*", "fft", or "linear")
/// @param fix Adjust length to match expected
/// @param scale Scale for energy preservation
/// @return Resampled signal
ArrayXr resample(const ArrayXr& y, Real orig_sr, Real target_sr,
                 const std::string& res_type = "kaiser_hq",
                 bool fix = true, bool scale = false);

/// Bounded-lag auto-correlation
/// @param y Input array
/// @param max_size Maximum correlation lag (nullopt for full)
/// @return Auto-correlation
ArrayXr autocorrelate(const ArrayXr& y, std::optional<int> max_size = std::nullopt);
ArrayXXr autocorrelate(const ArrayXXr& y, std::optional<int> max_size = std::nullopt,
                       int axis = -1);

/// Linear Prediction Coefficients via Burg's method
/// @param y Input signal
/// @param order LPC order
/// @return LP coefficients (order + 1)
ArrayXr lpc(const ArrayXr& y, int order);
ArrayXXr lpc(const ArrayXXr& y, int order, int axis = -1);

/// Find zero-crossings in a signal
/// @param y Input signal
/// @param threshold Threshold for considering a value as zero
/// @param ref_magnitude Reference magnitude for threshold scaling
/// @param pad Whether to consider first sample as zero-crossing
/// @param zero_pos Treat zero as positive
/// @return Boolean array indicating zero-crossings
Eigen::Array<bool, Eigen::Dynamic, 1>
zero_crossings(const ArrayXr& y,
               Real threshold = 1e-10,
               std::optional<Real> ref_magnitude = std::nullopt,
               bool pad = true,
               bool zero_pos = true);

Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
zero_crossings(const ArrayXXr& y,
               Real threshold = 1e-10,
               std::optional<Real> ref_magnitude = std::nullopt,
               bool pad = true,
               bool zero_pos = true,
               int axis = -1);

// ============================================================================
// Signal Generation
// ============================================================================

/// Generate a click track
/// @param times Times to place clicks (seconds)
/// @param sr Sample rate
/// @param click_freq Click frequency (Hz)
/// @param click_duration Click duration (seconds)
/// @param length Total signal length (samples)
/// @return Click track signal
ArrayXr clicks(const ArrayXr& times, Real sr = 22050,
               Real click_freq = 1000.0, Real click_duration = 0.1,
               std::optional<int> length = std::nullopt);

/// Generate a click track from frame indices
/// @param frames Frame indices
/// @param sr Sample rate
/// @param hop_length Hop length
/// @param click_freq Click frequency
/// @param click_duration Click duration
/// @param length Total signal length
/// @return Click track signal
ArrayXr clicks_frames(const std::vector<Eigen::Index>& frames, Real sr = 22050,
                      int hop_length = 512,
                      Real click_freq = 1000.0, Real click_duration = 0.1,
                      std::optional<int> length = std::nullopt);

/// Generate a pure tone (cosine) signal
/// @param frequency Frequency in Hz
/// @param sr Sample rate
/// @param length Number of samples (or use duration)
/// @param duration Duration in seconds (or use length)
/// @param phi Phase offset in radians
/// @return Tone signal
ArrayXr tone(Real frequency, Real sr = 22050,
             std::optional<int> length = std::nullopt,
             std::optional<Real> duration = std::nullopt,
             std::optional<Real> phi = std::nullopt);

/// Generate a chirp (sine-sweep) signal
/// @param fmin Starting frequency
/// @param fmax Ending frequency
/// @param sr Sample rate
/// @param length Number of samples
/// @param duration Duration in seconds
/// @param linear If true, linear sweep; else exponential
/// @param phi Phase offset
/// @return Chirp signal
ArrayXr chirp(Real fmin, Real fmax, Real sr = 22050,
              std::optional<int> length = std::nullopt,
              std::optional<Real> duration = std::nullopt,
              bool linear = false,
              std::optional<Real> phi = std::nullopt);

// ============================================================================
// Mu-law Compression/Expansion
// ============================================================================

/// Mu-law compression
/// @param x Input signal in range [-1, 1]
/// @param mu Compression parameter
/// @param quantize If true, quantize to 1+mu levels
/// @return Compressed signal
ArrayXr mu_compress(const ArrayXr& x, Real mu = 255, bool quantize = true);

/// Mu-law expansion (inverse of compression)
/// @param x Compressed signal
/// @param mu Compression parameter
/// @param quantize If true, input is quantized
/// @return Expanded signal
ArrayXr mu_expand(const ArrayXr& x, Real mu = 255, bool quantize = true);

} // namespace librosa
