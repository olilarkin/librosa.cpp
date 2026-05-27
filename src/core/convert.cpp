#include "librosa/core/convert.hpp"
#include "librosa/core/notation.hpp"
#include "librosa/util/exceptions.hpp"
#include <cmath>
#include <regex>
#include <map>

namespace librosa {

// ============================================================================
// Frame/Sample/Time Conversions
// ============================================================================

Eigen::Index frames_to_samples(Eigen::Index frames, int hop_length,
                               std::optional<int> n_fft) {
    Eigen::Index offset = n_fft ? (*n_fft / 2) : 0;
    return frames * hop_length + offset;
}

ArrayXr frames_to_samples(const ArrayXr& frames, int hop_length,
                          std::optional<int> n_fft) {
    Real offset = n_fft ? static_cast<Real>(*n_fft / 2) : 0.0;
    return (frames * static_cast<Real>(hop_length) + offset).floor();
}

Eigen::Index samples_to_frames(Eigen::Index samples, int hop_length,
                               std::optional<int> n_fft) {
    Eigen::Index offset = n_fft ? (*n_fft / 2) : 0;
    return (samples - offset) / hop_length;
}

ArrayXr samples_to_frames(const ArrayXr& samples, int hop_length,
                          std::optional<int> n_fft) {
    Real offset = n_fft ? static_cast<Real>(*n_fft / 2) : 0.0;
    return ((samples - offset) / static_cast<Real>(hop_length)).floor();
}

Real frames_to_time(Eigen::Index frames, Real sr, int hop_length,
                    std::optional<int> n_fft) {
    return static_cast<Real>(frames_to_samples(frames, hop_length, n_fft)) / sr;
}

ArrayXr frames_to_time(const ArrayXr& frames, Real sr, int hop_length,
                       std::optional<int> n_fft) {
    return frames_to_samples(frames, hop_length, n_fft) / sr;
}

Eigen::Index time_to_frames(Real times, Real sr, int hop_length,
                            std::optional<int> n_fft) {
    return samples_to_frames(static_cast<Eigen::Index>(times * sr), hop_length, n_fft);
}

ArrayXr time_to_frames(const ArrayXr& times, Real sr, int hop_length,
                       std::optional<int> n_fft) {
    return samples_to_frames((times * sr).floor(), hop_length, n_fft);
}

Eigen::Index time_to_samples(Real times, Real sr) {
    return static_cast<Eigen::Index>(times * sr);
}

ArrayXr time_to_samples(const ArrayXr& times, Real sr) {
    return (times * sr).floor();
}

Real samples_to_time(Eigen::Index samples, Real sr) {
    return static_cast<Real>(samples) / sr;
}

ArrayXr samples_to_time(const ArrayXr& samples, Real sr) {
    return samples / sr;
}

Eigen::Index blocks_to_frames(Eigen::Index blocks, int block_length) {
    return block_length * blocks;
}

ArrayXr blocks_to_frames(const ArrayXr& blocks, int block_length) {
    return blocks * static_cast<Real>(block_length);
}

Eigen::Index blocks_to_samples(Eigen::Index blocks, int block_length, int hop_length) {
    return frames_to_samples(blocks_to_frames(blocks, block_length), hop_length);
}

ArrayXr blocks_to_samples(const ArrayXr& blocks, int block_length, int hop_length) {
    return frames_to_samples(blocks_to_frames(blocks, block_length), hop_length);
}

Real blocks_to_time(Eigen::Index blocks, int block_length, int hop_length, Real sr) {
    return samples_to_time(blocks_to_samples(blocks, block_length, hop_length), sr);
}

ArrayXr blocks_to_time(const ArrayXr& blocks, int block_length, int hop_length, Real sr) {
    return samples_to_time(blocks_to_samples(blocks, block_length, hop_length), sr);
}

// ============================================================================
// Pitch/Frequency Conversions
// ============================================================================

Real midi_to_hz(Real note) {
    return 440.0 * std::pow(2.0, (note - 69.0) / 12.0);
}

ArrayXr midi_to_hz(const ArrayXr& notes) {
    return 440.0 * Eigen::pow(2.0, (notes - 69.0) / 12.0);
}

Real hz_to_midi(Real frequency) {
    return 12.0 * (std::log2(frequency) - std::log2(440.0)) + 69.0;
}

ArrayXr hz_to_midi(const ArrayXr& frequencies) {
    return 12.0 * (frequencies.log() / std::log(2.0) - std::log2(440.0)) + 69.0;
}

namespace {
    // Note name to semitone offset mapping
    const std::map<char, int> NOTE_MAP = {
        {'C', 0}, {'D', 2}, {'E', 4}, {'F', 5}, {'G', 7}, {'A', 9}, {'B', 11},
        {'c', 0}, {'d', 2}, {'e', 4}, {'f', 5}, {'g', 7}, {'a', 9}, {'b', 11}
    };

    // Note names for each semitone (using sharps)
    const std::vector<std::string> NOTE_NAMES_SHARP = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };

    const std::vector<std::string> NOTE_NAMES_SHARP_UNICODE = {
        "C", "C\u266F", "D", "D\u266F", "E", "F", "F\u266F", "G", "G\u266F", "A", "A\u266F", "B"
    };
}

Real note_to_midi(const std::string& note, bool round_midi) {
    if (note.empty()) {
        throw ParameterError("Empty note string");
    }

    // Parse note name
    auto it = NOTE_MAP.find(note[0]);
    if (it == NOTE_MAP.end()) {
        throw ParameterError("Invalid note name: " + note);
    }

    int pitch_class = it->second;
    int octave = 0;
    Real cents = 0.0;
    size_t i = 1;

    // Parse accidentals
    // Unicode ♯ (U+266F) is 3-byte UTF-8: 0xE2 0x99 0xAF
    // Unicode ♭ (U+266D) is 3-byte UTF-8: 0xE2 0x99 0xAD
    while (i < note.size()) {
        unsigned char c = static_cast<unsigned char>(note[i]);
        if (c == '#') {
            pitch_class += 1;
            ++i;
        } else if (i + 2 < note.size() && c == 0xE2 &&
                   static_cast<unsigned char>(note[i+1]) == 0x99 &&
                   static_cast<unsigned char>(note[i+2]) == 0xAF) {
            // Unicode sharp ♯
            pitch_class += 1;
            i += 3;
        } else if (c == 'b' || c == '!') {
            pitch_class -= 1;
            ++i;
        } else if (i + 2 < note.size() && c == 0xE2 &&
                   static_cast<unsigned char>(note[i+1]) == 0x99 &&
                   static_cast<unsigned char>(note[i+2]) == 0xAD) {
            // Unicode flat ♭
            pitch_class -= 1;
            i += 3;
        } else {
            break;
        }
    }

    // Parse octave
    bool has_octave = false;
    bool negative_octave = false;
    if (i < note.size() && (note[i] == '-' || std::isdigit(note[i]))) {
        has_octave = true;
        if (note[i] == '-') {
            negative_octave = true;
            ++i;
        }
        while (i < note.size() && std::isdigit(note[i])) {
            octave = octave * 10 + (note[i] - '0');
            ++i;
        }
        if (negative_octave) {
            octave = -octave;
        }
    }

    // Parse cents (e.g., +30 or -15)
    if (i < note.size() && (note[i] == '+' || note[i] == '-')) {
        bool negative_cents = (note[i] == '-');
        ++i;
        int cents_int = 0;
        while (i < note.size() && std::isdigit(note[i])) {
            cents_int = cents_int * 10 + (note[i] - '0');
            ++i;
        }
        cents = negative_cents ? -cents_int : cents_int;
    }

    // Check for unparsed characters
    if (i != note.size()) {
        throw ParameterError("Invalid note string: " + note);
    }

    // Compute MIDI number
    Real midi = static_cast<Real>((octave + 1) * 12 + pitch_class) + cents / 100.0;

    if (round_midi) {
        return std::round(midi);
    }
    return midi;
}

ArrayXr note_to_midi(const std::vector<std::string>& notes, bool round_midi) {
    ArrayXr result(notes.size());
    for (size_t i = 0; i < notes.size(); ++i) {
        result(i) = note_to_midi(notes[i], round_midi);
    }
    return result;
}

Real note_to_hz(const std::string& note) {
    return midi_to_hz(note_to_midi(note));
}

ArrayXr note_to_hz(const std::vector<std::string>& notes) {
    return midi_to_hz(note_to_midi(notes));
}

std::string midi_to_note(Real midi, bool octave, bool cents_flag, bool use_unicode) {
    if (cents_flag && !octave) {
        throw ParameterError("Cannot encode cents without octave information");
    }

    int note_num = static_cast<int>(std::round(midi));
    int note_cents = static_cast<int>(100.0 * (midi - note_num) + 0.5);

    int pitch_class = ((note_num % 12) + 12) % 12;  // Handle negative MIDI numbers
    int oct = note_num / 12 - 1;

    const auto& note_names = use_unicode ? NOTE_NAMES_SHARP_UNICODE : NOTE_NAMES_SHARP;
    std::string note = note_names[pitch_class];

    if (octave) {
        note += std::to_string(oct);
    }

    if (cents_flag && note_cents != 0) {
        char sign = note_cents >= 0 ? '+' : '-';
        note += sign + std::to_string(std::abs(note_cents));
    }

    return note;
}

std::vector<std::string> midi_to_note(const ArrayXr& midi, bool octave,
                                      bool cents_flag, bool use_unicode) {
    std::vector<std::string> result(midi.size());
    for (Eigen::Index i = 0; i < midi.size(); ++i) {
        result[i] = midi_to_note(midi(i), octave, cents_flag, use_unicode);
    }
    return result;
}

std::string hz_to_note(Real frequency, bool octave, bool cents_flag, bool use_unicode) {
    return midi_to_note(hz_to_midi(frequency), octave, cents_flag, use_unicode);
}

std::vector<std::string> hz_to_note(const ArrayXr& frequencies, bool octave,
                                    bool cents_flag, bool use_unicode) {
    return midi_to_note(hz_to_midi(frequencies), octave, cents_flag, use_unicode);
}

Real hz_to_mel(Real frequency, bool htk) {
    if (htk) {
        return 2595.0 * std::log10(1.0 + frequency / 700.0);
    }

    // Slaney formula
    const Real f_min = 0.0;
    const Real f_sp = 200.0 / 3.0;
    const Real min_log_hz = 1000.0;
    const Real min_log_mel = (min_log_hz - f_min) / f_sp;
    const Real logstep = std::log(6.4) / 27.0;

    if (frequency < min_log_hz) {
        return (frequency - f_min) / f_sp;
    }
    return min_log_mel + std::log(frequency / min_log_hz) / logstep;
}

ArrayXr hz_to_mel(const ArrayXr& frequencies, bool htk) {
    if (htk) {
        return 2595.0 * (1.0 + frequencies / 700.0).log10();
    }

    // Slaney formula
    const Real f_min = 0.0;
    const Real f_sp = 200.0 / 3.0;
    const Real min_log_hz = 1000.0;
    const Real min_log_mel = (min_log_hz - f_min) / f_sp;
    const Real logstep = std::log(6.4) / 27.0;

    ArrayXr mels(frequencies.size());
    for (Eigen::Index i = 0; i < frequencies.size(); ++i) {
        if (frequencies(i) < min_log_hz) {
            mels(i) = (frequencies(i) - f_min) / f_sp;
        } else {
            mels(i) = min_log_mel + std::log(frequencies(i) / min_log_hz) / logstep;
        }
    }
    return mels;
}

Real mel_to_hz(Real mel, bool htk) {
    if (htk) {
        return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    }

    // Slaney formula
    const Real f_min = 0.0;
    const Real f_sp = 200.0 / 3.0;
    const Real min_log_hz = 1000.0;
    const Real min_log_mel = (min_log_hz - f_min) / f_sp;
    const Real logstep = std::log(6.4) / 27.0;

    if (mel < min_log_mel) {
        return f_min + f_sp * mel;
    }
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
}

ArrayXr mel_to_hz(const ArrayXr& mels, bool htk) {
    if (htk) {
        return 700.0 * (Eigen::pow(10.0, mels / 2595.0) - 1.0);
    }

    // Slaney formula
    const Real f_min = 0.0;
    const Real f_sp = 200.0 / 3.0;
    const Real min_log_hz = 1000.0;
    const Real min_log_mel = (min_log_hz - f_min) / f_sp;
    const Real logstep = std::log(6.4) / 27.0;

    ArrayXr freqs(mels.size());
    for (Eigen::Index i = 0; i < mels.size(); ++i) {
        if (mels(i) < min_log_mel) {
            freqs(i) = f_min + f_sp * mels(i);
        } else {
            freqs(i) = min_log_hz * std::exp(logstep * (mels(i) - min_log_mel));
        }
    }
    return freqs;
}

Real hz_to_octs(Real frequency, Real tuning, int bins_per_octave) {
    Real A440 = 440.0 * std::pow(2.0, tuning / bins_per_octave);
    return std::log2(frequency / (A440 / 16.0));
}

ArrayXr hz_to_octs(const ArrayXr& frequencies, Real tuning, int bins_per_octave) {
    Real A440 = 440.0 * std::pow(2.0, tuning / bins_per_octave);
    return (frequencies / (A440 / 16.0)).log() / std::log(2.0);
}

Real octs_to_hz(Real oct, Real tuning, int bins_per_octave) {
    Real A440 = 440.0 * std::pow(2.0, tuning / bins_per_octave);
    return (A440 / 16.0) * std::pow(2.0, oct);
}

ArrayXr octs_to_hz(const ArrayXr& octs, Real tuning, int bins_per_octave) {
    Real A440 = 440.0 * std::pow(2.0, tuning / bins_per_octave);
    return (A440 / 16.0) * Eigen::pow(2.0, octs);
}

Real A4_to_tuning(Real A4, int bins_per_octave) {
    return bins_per_octave * (std::log2(A4) - std::log2(440.0));
}

ArrayXr A4_to_tuning(const ArrayXr& A4, int bins_per_octave) {
    return bins_per_octave * (A4.log() / std::log(2.0) - std::log2(440.0));
}

Real tuning_to_A4(Real tuning, int bins_per_octave) {
    return 440.0 * std::pow(2.0, tuning / bins_per_octave);
}

ArrayXr tuning_to_A4(const ArrayXr& tuning, int bins_per_octave) {
    return 440.0 * Eigen::pow(2.0, tuning / static_cast<Real>(bins_per_octave));
}

// ============================================================================
// Frequency Arrays
// ============================================================================

ArrayXr fft_frequencies(Real sr, int n_fft) {
    Eigen::Index n_bins = n_fft / 2 + 1;
    ArrayXr freqs(n_bins);
    for (Eigen::Index i = 0; i < n_bins; ++i) {
        freqs(i) = static_cast<Real>(i) * sr / n_fft;
    }
    return freqs;
}

ArrayXr cqt_frequencies(int n_bins, Real fmin, int bins_per_octave, Real tuning) {
    Real correction = std::pow(2.0, tuning / bins_per_octave);
    ArrayXr frequencies(n_bins);
    for (int i = 0; i < n_bins; ++i) {
        frequencies(i) = correction * fmin * std::pow(2.0, static_cast<Real>(i) / bins_per_octave);
    }
    return frequencies;
}

ArrayXr mel_frequencies(int n_mels, Real fmin, Real fmax, bool htk) {
    Real min_mel = hz_to_mel(fmin, htk);
    Real max_mel = hz_to_mel(fmax, htk);

    ArrayXr mels(n_mels);
    for (int i = 0; i < n_mels; ++i) {
        mels(i) = min_mel + (max_mel - min_mel) * static_cast<Real>(i) / (n_mels - 1);
    }

    return mel_to_hz(mels, htk);
}

ArrayXr tempo_frequencies(int n_bins, int hop_length, Real sr) {
    if (n_bins < 1) {
        throw ParameterError("n_bins must be positive");
    }

    ArrayXr freqs(n_bins);
    freqs(0) = std::numeric_limits<Real>::infinity();
    for (int i = 1; i < n_bins; ++i) {
        freqs(i) = 60.0 * sr / (hop_length * static_cast<Real>(i));
    }
    return freqs;
}

ArrayXr fourier_tempo_frequencies(Real sr, int win_length, int hop_length) {
    return fft_frequencies(sr * 60.0 / hop_length, win_length);
}

// ============================================================================
// Frequency Weighting
// ============================================================================

Real A_weighting(Real frequency, std::optional<Real> min_db) {
    Real f_sq = frequency * frequency;

    const Real c0 = 12194.217 * 12194.217;
    const Real c1 = 20.598997 * 20.598997;
    const Real c2 = 107.65265 * 107.65265;
    const Real c3 = 737.86223 * 737.86223;

    Real weight = 2.0 + 20.0 * (
        std::log10(c0) +
        2.0 * std::log10(f_sq) -
        std::log10(f_sq + c0) -
        std::log10(f_sq + c1) -
        0.5 * std::log10(f_sq + c2) -
        0.5 * std::log10(f_sq + c3)
    );

    if (min_db) {
        weight = std::max(*min_db, weight);
    }
    return weight;
}

ArrayXr A_weighting(const ArrayXr& frequencies, std::optional<Real> min_db) {
    ArrayXr f_sq = frequencies.square();

    const Real c0 = 12194.217 * 12194.217;
    const Real c1 = 20.598997 * 20.598997;
    const Real c2 = 107.65265 * 107.65265;
    const Real c3 = 737.86223 * 737.86223;

    ArrayXr weights = 2.0 + 20.0 * (
        std::log10(c0) +
        2.0 * f_sq.log10() -
        (f_sq + c0).log10() -
        (f_sq + c1).log10() -
        0.5 * (f_sq + c2).log10() -
        0.5 * (f_sq + c3).log10()
    );

    if (min_db) {
        weights = weights.max(*min_db);
    }
    return weights;
}

Real B_weighting(Real frequency, std::optional<Real> min_db) {
    Real f_sq = frequency * frequency;

    const Real c0 = 12194.217 * 12194.217;
    const Real c1 = 20.598997 * 20.598997;
    const Real c2 = 158.48932 * 158.48932;

    Real weight = 0.17 + 20.0 * (
        std::log10(c0) +
        1.5 * std::log10(f_sq) -
        std::log10(f_sq + c0) -
        std::log10(f_sq + c1) -
        0.5 * std::log10(f_sq + c2)
    );

    if (min_db) {
        weight = std::max(*min_db, weight);
    }
    return weight;
}

ArrayXr B_weighting(const ArrayXr& frequencies, std::optional<Real> min_db) {
    ArrayXr f_sq = frequencies.square();

    const Real c0 = 12194.217 * 12194.217;
    const Real c1 = 20.598997 * 20.598997;
    const Real c2 = 158.48932 * 158.48932;

    ArrayXr weights = 0.17 + 20.0 * (
        std::log10(c0) +
        1.5 * f_sq.log10() -
        (f_sq + c0).log10() -
        (f_sq + c1).log10() -
        0.5 * (f_sq + c2).log10()
    );

    if (min_db) {
        weights = weights.max(*min_db);
    }
    return weights;
}

Real C_weighting(Real frequency, std::optional<Real> min_db) {
    Real f_sq = frequency * frequency;

    const Real c0 = 12194.217 * 12194.217;
    const Real c1 = 20.598997 * 20.598997;

    Real weight = 0.062 + 20.0 * (
        std::log10(c0) +
        std::log10(f_sq) -
        std::log10(f_sq + c0) -
        std::log10(f_sq + c1)
    );

    if (min_db) {
        weight = std::max(*min_db, weight);
    }
    return weight;
}

ArrayXr C_weighting(const ArrayXr& frequencies, std::optional<Real> min_db) {
    ArrayXr f_sq = frequencies.square();

    const Real c0 = 12194.217 * 12194.217;
    const Real c1 = 20.598997 * 20.598997;

    ArrayXr weights = 0.062 + 20.0 * (
        std::log10(c0) +
        f_sq.log10() -
        (f_sq + c0).log10() -
        (f_sq + c1).log10()
    );

    if (min_db) {
        weights = weights.max(*min_db);
    }
    return weights;
}

Real D_weighting(Real frequency, std::optional<Real> min_db) {
    Real f_sq = frequency * frequency;

    const Real c0 = 8.3046305e-3 * 8.3046305e-3;
    const Real c1 = 1018.7 * 1018.7;
    const Real c2 = 1039.6 * 1039.6;
    const Real c3 = 3136.5 * 3136.5;
    const Real c4 = 3424.0 * 3424.0;
    const Real c5 = 282.7 * 282.7;
    const Real c6 = 1160.0 * 1160.0;

    Real weight = 20.0 * (
        0.5 * std::log10(f_sq) -
        std::log10(c0) +
        0.5 * (
            std::log10((c1 - f_sq) * (c1 - f_sq) + c2 * f_sq) -
            std::log10((c3 - f_sq) * (c3 - f_sq) + c4 * f_sq) -
            std::log10(f_sq + c5) -
            std::log10(f_sq + c6)
        )
    );

    if (min_db) {
        weight = std::max(*min_db, weight);
    }
    return weight;
}

ArrayXr D_weighting(const ArrayXr& frequencies, std::optional<Real> min_db) {
    ArrayXr f_sq = frequencies.square();

    const Real c0 = 8.3046305e-3 * 8.3046305e-3;
    const Real c1 = 1018.7 * 1018.7;
    const Real c2 = 1039.6 * 1039.6;
    const Real c3 = 3136.5 * 3136.5;
    const Real c4 = 3424.0 * 3424.0;
    const Real c5 = 282.7 * 282.7;
    const Real c6 = 1160.0 * 1160.0;

    ArrayXr weights(frequencies.size());
    for (Eigen::Index i = 0; i < frequencies.size(); ++i) {
        Real fs = f_sq(i);
        weights(i) = 20.0 * (
            0.5 * std::log10(fs) -
            std::log10(c0) +
            0.5 * (
                std::log10((c1 - fs) * (c1 - fs) + c2 * fs) -
                std::log10((c3 - fs) * (c3 - fs) + c4 * fs) -
                std::log10(fs + c5) -
                std::log10(fs + c6)
            )
        );
    }

    if (min_db) {
        weights = weights.max(*min_db);
    }
    return weights;
}

Real Z_weighting(Real /*frequency*/, std::optional<Real> /*min_db*/) {
    return 0.0;
}

ArrayXr Z_weighting(const ArrayXr& frequencies, std::optional<Real> /*min_db*/) {
    return ArrayXr::Zero(frequencies.size());
}

ArrayXr frequency_weighting(const ArrayXr& frequencies, WeightType kind,
                            std::optional<Real> min_db) {
    switch (kind) {
        case WeightType::A:
            return A_weighting(frequencies, min_db);
        case WeightType::B:
            return B_weighting(frequencies, min_db);
        case WeightType::C:
            return C_weighting(frequencies, min_db);
        case WeightType::D:
            return D_weighting(frequencies, min_db);
        case WeightType::Z:
            return Z_weighting(frequencies, min_db);
        default:
            throw ParameterError("Unknown weight type");
    }
}

// ============================================================================
// Svara Conversions
// ============================================================================

namespace {

const std::vector<std::string> SVARA_MAP_LONG = {
    "Sa", "re", "Re", "ga", "Ga", "ma", "Ma", "Pa", "dha", "Dha", "ni", "Ni"
};

const std::vector<std::string> SVARA_MAP_SHORT = {
    "S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"
};

std::string decorate_octave(const std::string& svara, int svara_num, bool use_unicode) {
    std::string result = svara;
    if (svara_num >= 12 && svara_num < 24) {
        // Upper octave: overdot
        if (use_unicode) {
            // Insert combining dot above (U+0307) after first character
            // First char may be multi-byte, but svara names are ASCII
            result = result.substr(0, 1) + "\u0307" + result.substr(1);
        } else {
            result += "'";
        }
    } else if (svara_num >= -12 && svara_num < 0) {
        // Lower octave: underdot
        if (use_unicode) {
            result = result.substr(0, 1) + "\u0323" + result.substr(1);
        } else {
            result += ",";
        }
    }
    return result;
}

} // anonymous namespace

std::string midi_to_svara_h(Real midi, Real Sa, bool abbr,
                             bool octave, bool use_unicode) {
    int svara_num = static_cast<int>(std::round(midi - Sa));

    const auto& svara_map = abbr ? SVARA_MAP_SHORT : SVARA_MAP_LONG;
    std::string svara = svara_map[((svara_num % 12) + 12) % 12];

    if (octave) {
        svara = decorate_octave(svara, svara_num, use_unicode);
    }

    return svara;
}

std::string midi_to_svara_c(Real midi, Real Sa, int mela, bool abbr,
                             bool octave, bool use_unicode) {
    int svara_num = static_cast<int>(std::round(midi - Sa));

    auto svara_map = mela_to_svara(mela, abbr, use_unicode);
    std::string svara = svara_map[((svara_num % 12) + 12) % 12];

    if (octave) {
        svara = decorate_octave(svara, svara_num, use_unicode);
    }

    return svara;
}

std::string midi_to_svara_c(Real midi, Real Sa, const std::string& mela, bool abbr,
                             bool octave, bool use_unicode) {
    int svara_num = static_cast<int>(std::round(midi - Sa));

    auto svara_map = mela_to_svara(mela, abbr, use_unicode);
    std::string svara = svara_map[((svara_num % 12) + 12) % 12];

    if (octave) {
        svara = decorate_octave(svara, svara_num, use_unicode);
    }

    return svara;
}

std::string hz_to_svara_h(Real frequency, Real Sa, bool abbr,
                           bool octave, bool use_unicode) {
    return midi_to_svara_h(hz_to_midi(frequency), hz_to_midi(Sa),
                            abbr, octave, use_unicode);
}

std::string hz_to_svara_c(Real frequency, Real Sa, int mela, bool abbr,
                           bool octave, bool use_unicode) {
    return midi_to_svara_c(hz_to_midi(frequency), hz_to_midi(Sa),
                            mela, abbr, octave, use_unicode);
}

std::string hz_to_svara_c(Real frequency, Real Sa, const std::string& mela, bool abbr,
                           bool octave, bool use_unicode) {
    return midi_to_svara_c(hz_to_midi(frequency), hz_to_midi(Sa),
                            mela, abbr, octave, use_unicode);
}

std::string note_to_svara_h(const std::string& note, const std::string& Sa,
                             bool abbr, bool octave, bool use_unicode) {
    return midi_to_svara_h(note_to_midi(note, false), note_to_midi(Sa),
                            abbr, octave, use_unicode);
}

std::string note_to_svara_c(const std::string& note, const std::string& Sa, int mela,
                             bool abbr, bool octave, bool use_unicode) {
    return midi_to_svara_c(note_to_midi(note, false), note_to_midi(Sa),
                            mela, abbr, octave, use_unicode);
}

std::string note_to_svara_c(const std::string& note, const std::string& Sa,
                             const std::string& mela, bool abbr,
                             bool octave, bool use_unicode) {
    return midi_to_svara_c(note_to_midi(note, false), note_to_midi(Sa),
                            mela, abbr, octave, use_unicode);
}

std::string hz_to_fjs(Real frequency, Real fmin, const std::string& unison,
                       bool use_unicode) {
    Real actual_fmin = fmin;
    if (actual_fmin <= 0) {
        actual_fmin = frequency;
    }
    std::string actual_unison = unison;
    if (actual_unison.empty()) {
        actual_unison = hz_to_note(actual_fmin, false, false, false);
    }

    Real interval = frequency / actual_fmin;
    return interval_to_fjs(interval, actual_unison, 65.0 / 63.0, use_unicode);
}

// ============================================================================
// Utility — samples_like / times_like
// ============================================================================

ArrayXr samples_like(const ArrayXXr& X, int hop_length,
                     std::optional<int> n_fft, int axis) {
    if (axis < 0) {
        axis = 2 + axis;
    }
    Eigen::Index n_frames = (axis == 0) ? X.rows() : X.cols();
    ArrayXr frames(n_frames);
    for (Eigen::Index i = 0; i < n_frames; ++i) {
        frames(i) = static_cast<Real>(i);
    }
    return frames_to_samples(frames, hop_length, n_fft);
}

ArrayXr times_like(const ArrayXXr& X, Real sr, int hop_length,
                   std::optional<int> n_fft, int axis) {
    return samples_like(X, hop_length, n_fft, axis) / sr;
}

} // namespace librosa
