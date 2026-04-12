#pragma once

#include "../types.hpp"
#include <optional>
#include <string>
#include <vector>

namespace librosa {

// ============================================================================
// Frame/Sample/Time Conversions
// ============================================================================

/// Convert frame indices to audio sample indices
/// @param frames Frame index or vector of frame indices
/// @param hop_length Number of samples between successive frames
/// @param n_fft Optional FFT window length (adds offset of n_fft/2)
/// @return Sample indices
Eigen::Index frames_to_samples(Eigen::Index frames, int hop_length = 512,
                               std::optional<int> n_fft = std::nullopt);
ArrayXr frames_to_samples(const ArrayXr& frames, int hop_length = 512,
                          std::optional<int> n_fft = std::nullopt);

/// Convert sample indices into STFT frames
/// @param samples Sample index or array
/// @param hop_length Number of samples between frames
/// @param n_fft Optional FFT window length
/// @return Frame indices
Eigen::Index samples_to_frames(Eigen::Index samples, int hop_length = 512,
                               std::optional<int> n_fft = std::nullopt);
ArrayXr samples_to_frames(const ArrayXr& samples, int hop_length = 512,
                          std::optional<int> n_fft = std::nullopt);

/// Convert frame indices to time (seconds)
/// @param frames Frame indices
/// @param sr Sample rate
/// @param hop_length Samples between frames
/// @param n_fft Optional FFT window length
/// @return Time in seconds
Real frames_to_time(Eigen::Index frames, Real sr = 22050, int hop_length = 512,
                    std::optional<int> n_fft = std::nullopt);
ArrayXr frames_to_time(const ArrayXr& frames, Real sr = 22050, int hop_length = 512,
                       std::optional<int> n_fft = std::nullopt);

/// Convert time values to frame indices
/// @param times Time values in seconds
/// @param sr Sample rate
/// @param hop_length Samples between frames
/// @param n_fft Optional FFT window length
/// @return Frame indices
Eigen::Index time_to_frames(Real times, Real sr = 22050, int hop_length = 512,
                            std::optional<int> n_fft = std::nullopt);
ArrayXr time_to_frames(const ArrayXr& times, Real sr = 22050, int hop_length = 512,
                       std::optional<int> n_fft = std::nullopt);

/// Convert time values to sample indices
/// @param times Time values in seconds
/// @param sr Sample rate
/// @return Sample indices
Eigen::Index time_to_samples(Real times, Real sr = 22050);
ArrayXr time_to_samples(const ArrayXr& times, Real sr = 22050);

/// Convert sample indices to time (seconds)
/// @param samples Sample indices
/// @param sr Sample rate
/// @return Time values in seconds
Real samples_to_time(Eigen::Index samples, Real sr = 22050);
ArrayXr samples_to_time(const ArrayXr& samples, Real sr = 22050);

/// Convert block indices to frame indices
Eigen::Index blocks_to_frames(Eigen::Index blocks, int block_length);
ArrayXr blocks_to_frames(const ArrayXr& blocks, int block_length);

/// Convert block indices to sample indices
Eigen::Index blocks_to_samples(Eigen::Index blocks, int block_length, int hop_length);
ArrayXr blocks_to_samples(const ArrayXr& blocks, int block_length, int hop_length);

/// Convert block indices to time (seconds)
Real blocks_to_time(Eigen::Index blocks, int block_length, int hop_length, Real sr);
ArrayXr blocks_to_time(const ArrayXr& blocks, int block_length, int hop_length, Real sr);

// ============================================================================
// Pitch/Frequency Conversions
// ============================================================================

/// Convert MIDI note number to frequency (Hz)
/// @param notes MIDI note number(s)
/// @return Frequency in Hz
Real midi_to_hz(Real note);
ArrayXr midi_to_hz(const ArrayXr& notes);

/// Convert frequency (Hz) to MIDI note number
/// @param frequencies Frequency in Hz
/// @return MIDI note number
Real hz_to_midi(Real frequency);
ArrayXr hz_to_midi(const ArrayXr& frequencies);

/// Convert note name to MIDI number
/// @param note Note name (e.g., "C4", "A#3", "Bb2")
/// @param round_midi If true, round to nearest integer
/// @return MIDI note number
Real note_to_midi(const std::string& note, bool round_midi = true);
ArrayXr note_to_midi(const std::vector<std::string>& notes, bool round_midi = true);

/// Convert note name to frequency (Hz)
/// @param note Note name
/// @return Frequency in Hz
Real note_to_hz(const std::string& note);
ArrayXr note_to_hz(const std::vector<std::string>& notes);

/// Convert MIDI note number to note name
/// @param midi MIDI note number
/// @param octave Include octave number
/// @param cents Include cent deviations
/// @param use_unicode Use Unicode accidentals (♯/♭ vs #/b)
/// @return Note name string
std::string midi_to_note(Real midi, bool octave = true, bool cents = false,
                         bool use_unicode = true);
std::vector<std::string> midi_to_note(const ArrayXr& midi, bool octave = true,
                                      bool cents = false, bool use_unicode = true);

/// Convert frequency (Hz) to note name
/// @param frequency Frequency in Hz
/// @return Note name string
std::string hz_to_note(Real frequency, bool octave = true, bool cents = false,
                       bool use_unicode = true);
std::vector<std::string> hz_to_note(const ArrayXr& frequencies, bool octave = true,
                                    bool cents = false, bool use_unicode = true);

/// Convert Hz to mel scale
/// @param frequencies Frequency in Hz
/// @param htk Use HTK formula (vs Slaney)
/// @return Mel value
Real hz_to_mel(Real frequency, bool htk = false);
ArrayXr hz_to_mel(const ArrayXr& frequencies, bool htk = false);

/// Convert mel scale to Hz
/// @param mels Mel values
/// @param htk Use HTK formula (vs Slaney)
/// @return Frequency in Hz
Real mel_to_hz(Real mel, bool htk = false);
ArrayXr mel_to_hz(const ArrayXr& mels, bool htk = false);

/// Convert Hz to octave number (relative to A)
/// @param frequencies Frequency in Hz
/// @param tuning Tuning deviation from A440
/// @param bins_per_octave Number of bins per octave
/// @return Octave numbers
Real hz_to_octs(Real frequency, Real tuning = 0.0, int bins_per_octave = 12);
ArrayXr hz_to_octs(const ArrayXr& frequencies, Real tuning = 0.0, int bins_per_octave = 12);

/// Convert octave numbers to Hz
/// @param octs Octave numbers
/// @param tuning Tuning deviation from A440
/// @param bins_per_octave Number of bins per octave
/// @return Frequency in Hz
Real octs_to_hz(Real oct, Real tuning = 0.0, int bins_per_octave = 12);
ArrayXr octs_to_hz(const ArrayXr& octs, Real tuning = 0.0, int bins_per_octave = 12);

/// Convert reference pitch frequency (A4) to tuning deviation
/// @param A4 Reference frequency for A4
/// @param bins_per_octave Number of bins per octave
/// @return Tuning deviation in fractional bins
Real A4_to_tuning(Real A4, int bins_per_octave = 12);
ArrayXr A4_to_tuning(const ArrayXr& A4, int bins_per_octave = 12);

/// Convert tuning deviation to reference pitch frequency (A4)
/// @param tuning Tuning deviation in fractional bins
/// @param bins_per_octave Number of bins per octave
/// @return Reference frequency for A4
Real tuning_to_A4(Real tuning, int bins_per_octave = 12);
ArrayXr tuning_to_A4(const ArrayXr& tuning, int bins_per_octave = 12);

// ============================================================================
// Frequency Arrays
// ============================================================================

/// Compute FFT bin center frequencies
/// @param sr Sample rate
/// @param n_fft FFT window size
/// @return Array of frequencies (0, sr/n_fft, 2*sr/n_fft, ..., sr/2)
ArrayXr fft_frequencies(Real sr = 22050, int n_fft = 2048);

/// Compute CQT bin center frequencies
/// @param n_bins Number of CQT bins
/// @param fmin Minimum frequency
/// @param bins_per_octave Bins per octave
/// @param tuning Tuning deviation
/// @return Array of CQT center frequencies
ArrayXr cqt_frequencies(int n_bins, Real fmin, int bins_per_octave = 12, Real tuning = 0.0);

/// Compute mel frequency array
/// @param n_mels Number of mel bins
/// @param fmin Minimum frequency
/// @param fmax Maximum frequency
/// @param htk Use HTK formula
/// @return Array of mel band center frequencies
ArrayXr mel_frequencies(int n_mels = 128, Real fmin = 0.0, Real fmax = 11025.0, bool htk = false);

/// Compute tempo frequencies for autocorrelation tempogram
/// @param n_bins Number of lag bins
/// @param hop_length Samples between frames
/// @param sr Sample rate
/// @return Array of tempo frequencies in BPM
ArrayXr tempo_frequencies(int n_bins, int hop_length = 512, Real sr = 22050);

/// Compute frequencies for Fourier tempogram
/// @param sr Sample rate
/// @param win_length Frames per analysis window
/// @param hop_length Samples between frames
/// @return Array of tempo frequencies in BPM
ArrayXr fourier_tempo_frequencies(Real sr = 22050, int win_length = 384, int hop_length = 512);

// ============================================================================
// Frequency Weighting
// ============================================================================

/// Compute A-weighting of frequencies
/// @param frequencies Frequencies in Hz
/// @param min_db Minimum weight threshold (nullptr for no clipping)
/// @return A-weighting values in dB
Real A_weighting(Real frequency, std::optional<Real> min_db = -80.0);
ArrayXr A_weighting(const ArrayXr& frequencies, std::optional<Real> min_db = -80.0);

/// Compute B-weighting of frequencies
Real B_weighting(Real frequency, std::optional<Real> min_db = -80.0);
ArrayXr B_weighting(const ArrayXr& frequencies, std::optional<Real> min_db = -80.0);

/// Compute C-weighting of frequencies
Real C_weighting(Real frequency, std::optional<Real> min_db = -80.0);
ArrayXr C_weighting(const ArrayXr& frequencies, std::optional<Real> min_db = -80.0);

/// Compute D-weighting of frequencies
Real D_weighting(Real frequency, std::optional<Real> min_db = -80.0);
ArrayXr D_weighting(const ArrayXr& frequencies, std::optional<Real> min_db = -80.0);

/// Compute Z-weighting (zero weighting, returns zeros)
Real Z_weighting(Real frequency, std::optional<Real> min_db = -80.0);
ArrayXr Z_weighting(const ArrayXr& frequencies, std::optional<Real> min_db = -80.0);

/// Weighting type enum
enum class WeightType {
    A, B, C, D, Z
};

/// Compute frequency weighting by type
ArrayXr frequency_weighting(const ArrayXr& frequencies, WeightType kind,
                            std::optional<Real> min_db = -80.0);

// ============================================================================
// Svara Conversions (Indian Music Notation)
// ============================================================================

/// Convert MIDI number to Hindustani svara
/// @param midi MIDI note number
/// @param Sa MIDI number of the reference Sa
/// @param abbr If true, return abbreviated names ('S', 'r', 'R', ...)
/// @param octave If true, decorate with octave marks
/// @param use_unicode Use Unicode combining marks for octave decoration
/// @return Svara name string
std::string midi_to_svara_h(Real midi, Real Sa, bool abbr = true,
                             bool octave = true, bool use_unicode = true);

/// Convert MIDI number to Carnatic svara within a melakarta raga
/// @param midi MIDI note number
/// @param Sa MIDI number of the reference Sa
/// @param mela Melakarta raga index (1-72) or name
/// @param abbr If true, return abbreviated names
/// @param octave If true, decorate with octave marks
/// @param use_unicode Use Unicode symbols
/// @return Svara name string
std::string midi_to_svara_c(Real midi, Real Sa, int mela, bool abbr = true,
                             bool octave = true, bool use_unicode = true);
std::string midi_to_svara_c(Real midi, Real Sa, const std::string& mela, bool abbr = true,
                             bool octave = true, bool use_unicode = true);

/// Convert frequency (Hz) to Hindustani svara
/// @param frequency Frequency in Hz
/// @param Sa Reference Sa frequency in Hz
std::string hz_to_svara_h(Real frequency, Real Sa, bool abbr = true,
                           bool octave = true, bool use_unicode = true);

/// Convert frequency (Hz) to Carnatic svara
/// @param frequency Frequency in Hz
/// @param Sa Reference Sa frequency in Hz
/// @param mela Melakarta raga index or name
std::string hz_to_svara_c(Real frequency, Real Sa, int mela, bool abbr = true,
                           bool octave = true, bool use_unicode = true);
std::string hz_to_svara_c(Real frequency, Real Sa, const std::string& mela, bool abbr = true,
                           bool octave = true, bool use_unicode = true);

/// Convert note name to Hindustani svara
/// @param note Western note name (e.g., "C4", "D#3")
/// @param Sa Note name of the reference Sa (e.g., "C4")
std::string note_to_svara_h(const std::string& note, const std::string& Sa,
                             bool abbr = true, bool octave = true, bool use_unicode = true);

/// Convert note name to Carnatic svara
/// @param note Western note name
/// @param Sa Note name of the reference Sa
/// @param mela Melakarta raga index or name
std::string note_to_svara_c(const std::string& note, const std::string& Sa, int mela,
                             bool abbr = true, bool octave = true, bool use_unicode = true);
std::string note_to_svara_c(const std::string& note, const std::string& Sa,
                             const std::string& mela, bool abbr = true,
                             bool octave = true, bool use_unicode = true);

/// Convert frequency (Hz) to Functional Just System notation
/// @param frequency Frequency in Hz
/// @param fmin Minimum frequency (unison). If <= 0, inferred from frequency.
/// @param unison Name of the unison note. If empty, inferred from fmin.
/// @param use_unicode Use Unicode accidentals and super/subscripts
/// @return Note name in FJS notation
std::string hz_to_fjs(Real frequency, Real fmin = 0, const std::string& unison = "",
                       bool use_unicode = true);

/// Generate sample indices like a reference array
/// @param X Reference array for shape
/// @param hop_length Samples between frames
/// @param n_fft FFT window length (for offset)
/// @param axis Axis along which X is time-aligned
/// @return Sample indices
ArrayXr samples_like(const ArrayXXr& X, int hop_length = 512,
                     std::optional<int> n_fft = std::nullopt, int axis = -1);

/// Generate time values like a reference array
/// @param X Reference array for shape
/// @param sr Sample rate
/// @param hop_length Samples between frames
/// @param n_fft FFT window length (for offset)
/// @param axis Axis along which X is time-aligned
/// @return Time values in seconds
ArrayXr times_like(const ArrayXXr& X, Real sr = 22050, int hop_length = 512,
                   std::optional<int> n_fft = std::nullopt, int axis = -1);

} // namespace librosa
