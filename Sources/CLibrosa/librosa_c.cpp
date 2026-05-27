#include "librosa_c.h"

#include <Eigen/Eigenvalues>

#include <librosa/beat.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/core/constantq.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/decompose.hpp>
#include <librosa/effects.hpp>
#include <librosa/feature/spectral.hpp>
#include <librosa/onset.hpp>
#include <librosa/segment.hpp>
#include <librosa/sequence.hpp>
#include <librosa/util/utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

void clear_error() {
    g_last_error.clear();
}

int fail_with(const std::string& message) {
    g_last_error = message;
    return LIBROSA_STATUS_ERROR;
}

int fail_current_exception() {
    try {
        throw;
    } catch (const std::exception& error) {
        return fail_with(error.what());
    } catch (...) {
        return fail_with("Unknown librosa error");
    }
}

bool validate_out(const void *out, const char *name) {
    if (!out) {
        g_last_error = std::string(name) + " must not be null";
        return false;
    }
    return true;
}

bool validate_input(const double *data, int64_t count) {
    if (count < 0) {
        g_last_error = "input count must be non-negative";
        return false;
    }
    if (count > 0 && !data) {
        g_last_error = "input data must not be null when count is positive";
        return false;
    }
    return true;
}

bool validate_matrix_input(const double *data, int64_t rows, int64_t columns) {
    if (rows < 0 || columns < 0) {
        g_last_error = "matrix shape must be non-negative";
        return false;
    }

    const int64_t count = rows * columns;
    if (count > 0 && !data) {
        g_last_error = "matrix data must not be null when shape is non-empty";
        return false;
    }
    return true;
}

bool validate_index_input(const int64_t *data, int64_t count) {
    if (count < 0) {
        g_last_error = "index count must be non-negative";
        return false;
    }
    if (count > 0 && !data) {
        g_last_error = "index data must not be null when count is positive";
        return false;
    }
    return true;
}

librosa::ArrayXr make_array(const double *data, int64_t count) {
    librosa::ArrayXr result(static_cast<Eigen::Index>(count));
    for (int64_t index = 0; index < count; ++index) {
        result(static_cast<Eigen::Index>(index)) = data[index];
    }
    return result;
}

librosa::ArrayXXr make_matrix(const double *data, int64_t rows, int64_t columns) {
    librosa::ArrayXXr result(static_cast<Eigen::Index>(rows),
                             static_cast<Eigen::Index>(columns));
    int64_t offset = 0;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < columns; ++column) {
            result(static_cast<Eigen::Index>(row),
                   static_cast<Eigen::Index>(column)) = data[offset++];
        }
    }
    return result;
}

std::vector<Eigen::Index> make_indices(const int64_t *data, int64_t count) {
    std::vector<Eigen::Index> indices;
    indices.reserve(static_cast<size_t>(count));
    for (int64_t index = 0; index < count; ++index) {
        indices.push_back(static_cast<Eigen::Index>(data[index]));
    }
    return indices;
}

int copy_vector(const librosa::ArrayXr& source, LibrosaVector *out) {
    if (!validate_out(out, "out")) {
        return LIBROSA_STATUS_ERROR;
    }

    out->count = static_cast<int64_t>(source.size());
    out->data = nullptr;

    if (out->count == 0) {
        return LIBROSA_STATUS_OK;
    }

    out->data = static_cast<double*>(
        std::malloc(static_cast<size_t>(out->count) * sizeof(double)));
    if (!out->data) {
        out->count = 0;
        return fail_with("failed to allocate output vector");
    }

    for (Eigen::Index index = 0; index < source.size(); ++index) {
        out->data[index] = source(index);
    }

    return LIBROSA_STATUS_OK;
}

int copy_matrix(const librosa::ArrayXXr& source, LibrosaMatrix *out) {
    if (!validate_out(out, "out")) {
        return LIBROSA_STATUS_ERROR;
    }

    out->rows = static_cast<int64_t>(source.rows());
    out->columns = static_cast<int64_t>(source.cols());
    out->data = nullptr;

    const int64_t count = out->rows * out->columns;
    if (count == 0) {
        return LIBROSA_STATUS_OK;
    }

    out->data = static_cast<double*>(
        std::malloc(static_cast<size_t>(count) * sizeof(double)));
    if (!out->data) {
        out->rows = 0;
        out->columns = 0;
        return fail_with("failed to allocate output matrix");
    }

    int64_t offset = 0;
    for (Eigen::Index row = 0; row < source.rows(); ++row) {
        for (Eigen::Index column = 0; column < source.cols(); ++column) {
            out->data[offset++] = source(row, column);
        }
    }

    return LIBROSA_STATUS_OK;
}

int copy_indices(const std::vector<Eigen::Index>& source, LibrosaIndexVector *out) {
    if (!validate_out(out, "out")) {
        return LIBROSA_STATUS_ERROR;
    }

    out->count = static_cast<int64_t>(source.size());
    out->data = nullptr;

    if (out->count == 0) {
        return LIBROSA_STATUS_OK;
    }

    out->data = static_cast<int64_t*>(
        std::malloc(static_cast<size_t>(out->count) * sizeof(int64_t)));
    if (!out->data) {
        out->count = 0;
        return fail_with("failed to allocate output index vector");
    }

    for (size_t index = 0; index < source.size(); ++index) {
        out->data[index] = static_cast<int64_t>(source[index]);
    }

    return LIBROSA_STATUS_OK;
}

int copy_path(const std::vector<std::pair<int, int>>& source, LibrosaIndexVector *out) {
    std::vector<Eigen::Index> flattened;
    flattened.reserve(source.size() * 2);
    for (const auto& [first, second] : source) {
        flattened.push_back(first);
        flattened.push_back(second);
    }
    return copy_indices(flattened, out);
}

int copy_audio(const librosa::AudioData& source, LibrosaAudioData *out) {
    if (!validate_out(out, "out")) {
        return LIBROSA_STATUS_ERROR;
    }

    out->channels = static_cast<int64_t>(source.samples.rows());
    out->samples = static_cast<int64_t>(source.samples.cols());
    out->sample_rate = source.sample_rate;
    out->data = nullptr;

    const int64_t count = out->channels * out->samples;
    if (count == 0) {
        return LIBROSA_STATUS_OK;
    }

    out->data = static_cast<double*>(
        std::malloc(static_cast<size_t>(count) * sizeof(double)));
    if (!out->data) {
        out->channels = 0;
        out->samples = 0;
        return fail_with("failed to allocate output audio data");
    }

    int64_t offset = 0;
    for (Eigen::Index channel = 0; channel < source.samples.rows(); ++channel) {
        for (Eigen::Index sample = 0; sample < source.samples.cols(); ++sample) {
            out->data[offset++] = source.samples(channel, sample);
        }
    }

    return LIBROSA_STATUS_OK;
}

template <typename Func>
int run(Func&& func) {
    clear_error();
    try {
        return func();
    } catch (...) {
        return fail_current_exception();
    }
}

std::optional<librosa::Real> optional_real(int has_value, double value) {
    if (has_value) {
        return value;
    }
    return std::nullopt;
}

std::optional<int> optional_int(int has_value, int64_t value) {
    if (has_value) {
        return static_cast<int>(value);
    }
    return std::nullopt;
}

std::string c_string_or_default(const char *value, const char *default_value) {
    if (value) {
        return value;
    }
    return default_value;
}

librosa::WeightType weight_type_from_string(const char *value) {
    std::string kind = c_string_or_default(value, "A");
    if (kind == "A" || kind == "a") return librosa::WeightType::A;
    if (kind == "B" || kind == "b") return librosa::WeightType::B;
    if (kind == "C" || kind == "c") return librosa::WeightType::C;
    if (kind == "D" || kind == "d") return librosa::WeightType::D;
    if (kind == "Z" || kind == "z") return librosa::WeightType::Z;
    throw librosa::ParameterError("Unsupported weighting type: " + kind);
}

} // namespace

extern "C" {

const char *librosa_last_error_message(void) {
    return g_last_error.c_str();
}

void librosa_clear_error(void) {
    clear_error();
}

void librosa_vector_free(LibrosaVector *vector) {
    if (!vector) {
        return;
    }
    std::free(vector->data);
    vector->data = nullptr;
    vector->count = 0;
}

void librosa_matrix_free(LibrosaMatrix *matrix) {
    if (!matrix) {
        return;
    }
    std::free(matrix->data);
    matrix->data = nullptr;
    matrix->rows = 0;
    matrix->columns = 0;
}

void librosa_index_vector_free(LibrosaIndexVector *vector) {
    if (!vector) {
        return;
    }
    std::free(vector->data);
    vector->data = nullptr;
    vector->count = 0;
}

void librosa_audio_data_free(LibrosaAudioData *audio) {
    if (!audio) {
        return;
    }
    std::free(audio->data);
    audio->data = nullptr;
    audio->channels = 0;
    audio->samples = 0;
    audio->sample_rate = 0;
}

void librosa_beat_track_result_free(LibrosaBeatTrackResult *result) {
    if (!result) {
        return;
    }
    librosa_index_vector_free(&result->beats);
    result->tempo = 0;
}

void librosa_trim_result_free(LibrosaTrimResult *result) {
    if (!result) {
        return;
    }
    librosa_vector_free(&result->audio);
    result->start = 0;
    result->end = 0;
}

void librosa_dtw_result_free(LibrosaDTWResult *result) {
    if (!result) {
        return;
    }
    librosa_matrix_free(&result->cost);
    librosa_index_vector_free(&result->path);
}

int librosa_audio_info(const char *path, LibrosaAudioFileInfo *out_info) {
    return run([&]() {
        if (!validate_out(path, "path") || !validate_out(out_info, "out_info")) {
            return LIBROSA_STATUS_ERROR;
        }

        librosa::AudioFileInfo info = librosa::get_audio_info(path);
        out_info->samples = static_cast<int64_t>(info.samples);
        out_info->sample_rate = info.sample_rate;
        out_info->channels = static_cast<int32_t>(info.channels);
        out_info->duration = info.duration;
        return LIBROSA_STATUS_OK;
    });
}

int librosa_load(const char *path,
                 int has_target_sample_rate,
                 double target_sample_rate,
                 int mono,
                 double offset,
                 int has_duration,
                 double duration,
                 LibrosaAudioData *out_audio) {
    return run([&]() {
        if (!validate_out(path, "path") || !validate_out(out_audio, "out_audio")) {
            return LIBROSA_STATUS_ERROR;
        }

        librosa::AudioData audio = librosa::load(
            path,
            optional_real(has_target_sample_rate, target_sample_rate),
            mono != 0,
            offset,
            optional_real(has_duration, duration));
        return copy_audio(audio, out_audio);
    });
}

int librosa_frames_to_samples(int64_t frames,
                              int hop_length,
                              int has_n_fft,
                              int n_fft,
                              int64_t *out_samples) {
    return run([&]() {
        if (!validate_out(out_samples, "out_samples")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_samples = static_cast<int64_t>(
            librosa::frames_to_samples(static_cast<Eigen::Index>(frames),
                                       hop_length,
                                       optional_int(has_n_fft, n_fft)));
        return LIBROSA_STATUS_OK;
    });
}

int librosa_frames_to_samples_vector(const double *frames,
                                     int64_t count,
                                     int hop_length,
                                     int has_n_fft,
                                     int n_fft,
                                     LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(frames, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::frames_to_samples(make_array(frames, count),
                                                      hop_length,
                                                      optional_int(has_n_fft, n_fft)),
                           out);
    });
}

int librosa_samples_to_frames(int64_t samples,
                              int hop_length,
                              int has_n_fft,
                              int n_fft,
                              int64_t *out_frames) {
    return run([&]() {
        if (!validate_out(out_frames, "out_frames")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_frames = static_cast<int64_t>(
            librosa::samples_to_frames(static_cast<Eigen::Index>(samples),
                                       hop_length,
                                       optional_int(has_n_fft, n_fft)));
        return LIBROSA_STATUS_OK;
    });
}

int librosa_samples_to_frames_vector(const double *samples,
                                     int64_t count,
                                     int hop_length,
                                     int has_n_fft,
                                     int n_fft,
                                     LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(samples, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::samples_to_frames(make_array(samples, count),
                                                      hop_length,
                                                      optional_int(has_n_fft, n_fft)),
                           out);
    });
}

int librosa_frames_to_time(int64_t frames,
                           double sample_rate,
                           int hop_length,
                           int has_n_fft,
                           int n_fft,
                           double *out_time) {
    return run([&]() {
        if (!validate_out(out_time, "out_time")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_time = librosa::frames_to_time(static_cast<Eigen::Index>(frames),
                                            sample_rate,
                                            hop_length,
                                            optional_int(has_n_fft, n_fft));
        return LIBROSA_STATUS_OK;
    });
}

int librosa_frames_to_time_vector(const double *frames,
                                  int64_t count,
                                  double sample_rate,
                                  int hop_length,
                                  int has_n_fft,
                                  int n_fft,
                                  LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(frames, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::frames_to_time(make_array(frames, count),
                                                   sample_rate,
                                                   hop_length,
                                                   optional_int(has_n_fft, n_fft)),
                           out);
    });
}

int librosa_time_to_frames(double time,
                           double sample_rate,
                           int hop_length,
                           int has_n_fft,
                           int n_fft,
                           int64_t *out_frames) {
    return run([&]() {
        if (!validate_out(out_frames, "out_frames")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_frames = static_cast<int64_t>(
            librosa::time_to_frames(time, sample_rate, hop_length, optional_int(has_n_fft, n_fft)));
        return LIBROSA_STATUS_OK;
    });
}

int librosa_time_to_frames_vector(const double *times,
                                  int64_t count,
                                  double sample_rate,
                                  int hop_length,
                                  int has_n_fft,
                                  int n_fft,
                                  LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(times, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::time_to_frames(make_array(times, count),
                                                   sample_rate,
                                                   hop_length,
                                                   optional_int(has_n_fft, n_fft)),
                           out);
    });
}

int librosa_time_to_samples(double time, double sample_rate, int64_t *out_samples) {
    return run([&]() {
        if (!validate_out(out_samples, "out_samples")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_samples = static_cast<int64_t>(librosa::time_to_samples(time, sample_rate));
        return LIBROSA_STATUS_OK;
    });
}

int librosa_time_to_samples_vector(const double *times,
                                   int64_t count,
                                   double sample_rate,
                                   LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(times, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::time_to_samples(make_array(times, count), sample_rate), out);
    });
}

int librosa_samples_to_time(int64_t samples, double sample_rate, double *out_time) {
    return run([&]() {
        if (!validate_out(out_time, "out_time")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_time = librosa::samples_to_time(static_cast<Eigen::Index>(samples), sample_rate);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_samples_to_time_vector(const double *samples,
                                   int64_t count,
                                   double sample_rate,
                                   LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(samples, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::samples_to_time(make_array(samples, count), sample_rate), out);
    });
}

int librosa_midi_to_hz(double midi, double *out_hz) {
    return run([&]() {
        if (!validate_out(out_hz, "out_hz")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_hz = librosa::midi_to_hz(midi);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_midi_to_hz_vector(const double *midi, int64_t count, LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(midi, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::midi_to_hz(make_array(midi, count)), out);
    });
}

int librosa_hz_to_midi(double hz, double *out_midi) {
    return run([&]() {
        if (!validate_out(out_midi, "out_midi")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_midi = librosa::hz_to_midi(hz);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_hz_to_midi_vector(const double *hz, int64_t count, LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(hz, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::hz_to_midi(make_array(hz, count)), out);
    });
}

int librosa_hz_to_mel(double hz, int htk, double *out_mel) {
    return run([&]() {
        if (!validate_out(out_mel, "out_mel")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_mel = librosa::hz_to_mel(hz, htk != 0);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_hz_to_mel_vector(const double *hz, int64_t count, int htk, LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(hz, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::hz_to_mel(make_array(hz, count), htk != 0), out);
    });
}

int librosa_mel_to_hz(double mel, int htk, double *out_hz) {
    return run([&]() {
        if (!validate_out(out_hz, "out_hz")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_hz = librosa::mel_to_hz(mel, htk != 0);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_mel_to_hz_vector(const double *mel, int64_t count, int htk, LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(mel, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::mel_to_hz(make_array(mel, count), htk != 0), out);
    });
}

int librosa_note_to_midi(const char *note, int round_midi, double *out_midi) {
    return run([&]() {
        if (!validate_out(note, "note") || !validate_out(out_midi, "out_midi")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_midi = librosa::note_to_midi(note, round_midi != 0);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_note_to_hz(const char *note, double *out_hz) {
    return run([&]() {
        if (!validate_out(note, "note") || !validate_out(out_hz, "out_hz")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_hz = librosa::note_to_hz(note);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_hz_to_octs(double hz, double tuning, int bins_per_octave, double *out_octs) {
    return run([&]() {
        if (!validate_out(out_octs, "out_octs")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_octs = librosa::hz_to_octs(hz, tuning, bins_per_octave);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_hz_to_octs_vector(const double *hz,
                              int64_t count,
                              double tuning,
                              int bins_per_octave,
                              LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(hz, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::hz_to_octs(make_array(hz, count), tuning, bins_per_octave), out);
    });
}

int librosa_octs_to_hz(double octs, double tuning, int bins_per_octave, double *out_hz) {
    return run([&]() {
        if (!validate_out(out_hz, "out_hz")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_hz = librosa::octs_to_hz(octs, tuning, bins_per_octave);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_octs_to_hz_vector(const double *octs,
                              int64_t count,
                              double tuning,
                              int bins_per_octave,
                              LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(octs, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::octs_to_hz(make_array(octs, count), tuning, bins_per_octave), out);
    });
}

int librosa_a4_to_tuning(double a4, int bins_per_octave, double *out_tuning) {
    return run([&]() {
        if (!validate_out(out_tuning, "out_tuning")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_tuning = librosa::A4_to_tuning(a4, bins_per_octave);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_a4_to_tuning_vector(const double *a4,
                                int64_t count,
                                int bins_per_octave,
                                LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(a4, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::A4_to_tuning(make_array(a4, count), bins_per_octave), out);
    });
}

int librosa_tuning_to_a4(double tuning, int bins_per_octave, double *out_a4) {
    return run([&]() {
        if (!validate_out(out_a4, "out_a4")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_a4 = librosa::tuning_to_A4(tuning, bins_per_octave);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_tuning_to_a4_vector(const double *tuning,
                                int64_t count,
                                int bins_per_octave,
                                LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(tuning, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::tuning_to_A4(make_array(tuning, count), bins_per_octave), out);
    });
}

int librosa_fft_frequencies(double sample_rate, int n_fft, LibrosaVector *out) {
    return run([&]() {
        return copy_vector(librosa::fft_frequencies(sample_rate, n_fft), out);
    });
}

int librosa_cqt_frequencies(int n_bins,
                            double fmin,
                            int bins_per_octave,
                            double tuning,
                            LibrosaVector *out) {
    return run([&]() {
        return copy_vector(librosa::cqt_frequencies(n_bins, fmin, bins_per_octave, tuning), out);
    });
}

int librosa_mel_frequencies(int n_mels, double fmin, double fmax, int htk, LibrosaVector *out) {
    return run([&]() {
        return copy_vector(librosa::mel_frequencies(n_mels, fmin, fmax, htk != 0), out);
    });
}

int librosa_tempo_frequencies(int n_bins,
                              int hop_length,
                              double sample_rate,
                              LibrosaVector *out) {
    return run([&]() {
        if (n_bins < 1) {
            return fail_with("n_bins must be positive");
        }
        return copy_vector(librosa::tempo_frequencies(n_bins, hop_length, sample_rate), out);
    });
}

int librosa_fourier_tempo_frequencies(double sample_rate,
                                      int win_length,
                                      int hop_length,
                                      LibrosaVector *out) {
    return run([&]() {
        return copy_vector(librosa::fourier_tempo_frequencies(sample_rate, win_length, hop_length), out);
    });
}

int librosa_frequency_weighting(const double *frequencies,
                                int64_t count,
                                const char *kind,
                                int has_min_db,
                                double min_db,
                                LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(frequencies, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(librosa::frequency_weighting(make_array(frequencies, count),
                                                        weight_type_from_string(kind),
                                                        optional_real(has_min_db, min_db)),
                           out);
    });
}

int librosa_tone(double frequency,
                 double sample_rate,
                 int has_length,
                 int64_t length,
                 int has_duration,
                 double duration,
                 int has_phi,
                 double phi,
                 LibrosaVector *out) {
    return run([&]() {
        librosa::ArrayXr result = librosa::tone(
            frequency,
            sample_rate,
            optional_int(has_length, length),
            optional_real(has_duration, duration),
            optional_real(has_phi, phi));
        return copy_vector(result, out);
    });
}

int librosa_chirp(double fmin,
                  double fmax,
                  double sample_rate,
                  int has_length,
                  int64_t length,
                  int has_duration,
                  double duration,
                  int linear,
                  int has_phi,
                  double phi,
                  LibrosaVector *out) {
    return run([&]() {
        librosa::ArrayXr result = librosa::chirp(
            fmin,
            fmax,
            sample_rate,
            optional_int(has_length, length),
            optional_real(has_duration, duration),
            linear != 0,
            optional_real(has_phi, phi));
        return copy_vector(result, out);
    });
}

int librosa_resample(const double *y,
                     int64_t count,
                     double original_sample_rate,
                     double target_sample_rate,
                     const char *res_type,
                     int fix,
                     int scale,
                     LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(
            librosa::resample(make_array(y, count),
                              original_sample_rate,
                              target_sample_rate,
                              c_string_or_default(res_type, "kaiser_hq"),
                              fix != 0,
                              scale != 0),
            out);
    });
}

int librosa_stft_magnitude(const double *y,
                           int64_t count,
                           int n_fft,
                           int hop_length,
                           LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        librosa::ArrayXXc spectrum = librosa::stft(make_array(y, count),
                                                   n_fft,
                                                   hop_length,
                                                   std::nullopt,
                                                   librosa::WindowType::Hann,
                                                   true,
                                                   librosa::PadMode::Constant);
        return copy_matrix(librosa::magnitude(spectrum), out);
    });
}

int librosa_amplitude_to_db(const double *s,
                            int64_t rows,
                            int64_t columns,
                            double ref,
                            double amin,
                            int has_top_db,
                            double top_db,
                            LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_matrix_input(s, rows, columns)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::amplitude_to_db(make_matrix(s, rows, columns),
                                     ref,
                                     amin,
                                     optional_real(has_top_db, top_db)),
            out);
    });
}

int librosa_power_to_db(const double *s,
                        int64_t rows,
                        int64_t columns,
                        double ref,
                        double amin,
                        int has_top_db,
                        double top_db,
                        LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_matrix_input(s, rows, columns)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::power_to_db(make_matrix(s, rows, columns),
                                 ref,
                                 amin,
                                 optional_real(has_top_db, top_db)),
            out);
    });
}

int librosa_hpss(const double *s,
                 int64_t rows,
                 int64_t columns,
                 int kernel_size,
                 double power,
                 int mask,
                 double margin,
                 LibrosaMatrix *out_harmonic,
                 LibrosaMatrix *out_percussive) {
    return run([&]() {
        if (!validate_matrix_input(s, rows, columns)) {
            return LIBROSA_STATUS_ERROR;
        }

        auto [harmonic, percussive] = librosa::decompose::hpss(
            make_matrix(s, rows, columns),
            kernel_size,
            power,
            mask != 0,
            margin);

        int status = copy_matrix(harmonic, out_harmonic);
        if (status != LIBROSA_STATUS_OK) {
            return status;
        }

        status = copy_matrix(percussive, out_percussive);
        if (status != LIBROSA_STATUS_OK) {
            librosa_matrix_free(out_harmonic);
            return status;
        }

        return LIBROSA_STATUS_OK;
    });
}

int librosa_cqt_magnitude(const double *y,
                          int64_t count,
                          double sample_rate,
                          int hop_length,
                          int has_fmin,
                          double fmin,
                          int n_bins,
                          int bins_per_octave,
                          LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        librosa::ArrayXXc cqt = librosa::cqt(make_array(y, count),
                                             sample_rate,
                                             hop_length,
                                             optional_real(has_fmin, fmin),
                                             n_bins,
                                             bins_per_octave);
        return copy_matrix(librosa::magnitude(cqt), out);
    });
}

int librosa_chroma_cqt(const double *y,
                       int64_t count,
                       double sample_rate,
                       int hop_length,
                       int has_fmin,
                       double fmin,
                       int n_chroma,
                       int n_octaves,
                       int bins_per_octave,
                       LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::chroma_cqt(make_array(y, count),
                                         sample_rate,
                                         hop_length,
                                         optional_real(has_fmin, fmin),
                                         std::numeric_limits<librosa::Real>::infinity(),
                                         0.0,
                                         std::nullopt,
                                         n_chroma,
                                         n_octaves,
                                         bins_per_octave),
            out);
    });
}

int librosa_harmonic_effect(const double *y,
                            int64_t count,
                            int kernel_size,
                            double power,
                            int mask,
                            double margin,
                            int n_fft,
                            int hop_length,
                            LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(
            librosa::effects::harmonic(make_array(y, count),
                                       kernel_size,
                                       power,
                                       mask != 0,
                                       margin,
                                       n_fft,
                                       hop_length),
            out);
    });
}

int librosa_nn_filter(const double *s,
                      int64_t rows,
                      int64_t columns,
                      const char *metric,
                      int aggregate_median,
                      int k,
                      int width,
                      LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_matrix_input(s, rows, columns)) {
            return LIBROSA_STATUS_ERROR;
        }
        librosa::ArrayXXr input = make_matrix(s, rows, columns);
        librosa::SparseMatrixXr rec = librosa::segment::recurrence_matrix_sparse(
            input,
            k,
            width,
            c_string_or_default(metric, "euclidean"));
        return copy_matrix(
            librosa::decompose::nn_filter(
                input,
                rec,
                aggregate_median != 0 ? librosa::AggregateFunc::Median : librosa::AggregateFunc::Mean),
            out);
    });
}

int librosa_softmask(const double *x,
                     const double *x_ref,
                     int64_t rows,
                     int64_t columns,
                     double power,
                     int split_zeros,
                     LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_matrix_input(x, rows, columns) ||
            !validate_matrix_input(x_ref, rows, columns)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::util::softmask(make_matrix(x, rows, columns),
                                    make_matrix(x_ref, rows, columns),
                                    power,
                                    split_zeros != 0),
            out);
    });
}

int librosa_median_filter(const double *s,
                          int64_t rows,
                          int64_t columns,
                          int size_rows,
                          int size_columns,
                          LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_matrix_input(s, rows, columns)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::decompose::median_filter_2d(make_matrix(s, rows, columns),
                                                 {size_rows, size_columns}),
            out);
    });
}

int librosa_sync(const double *s,
                 int64_t rows,
                 int64_t columns,
                 const int64_t *indices,
                 int64_t index_count,
                 int aggregate_median,
                 int pad,
                 int axis,
                 LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_matrix_input(s, rows, columns) ||
            !validate_index_input(indices, index_count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::util::sync(make_matrix(s, rows, columns),
                                make_indices(indices, index_count),
                                aggregate_median != 0 ? librosa::AggregateFunc::Median : librosa::AggregateFunc::Mean,
                                pad != 0,
                                axis),
            out);
    });
}

int librosa_recurrence_matrix(const double *s,
                              int64_t rows,
                              int64_t columns,
                              int k,
                              int width,
                              const char *metric,
                              int sym,
                              const char *mode,
                              double bandwidth,
                              int self,
                              LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_matrix_input(s, rows, columns)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::segment::recurrence_matrix(make_matrix(s, rows, columns),
                                                k,
                                                width,
                                                c_string_or_default(metric, "euclidean"),
                                                sym != 0,
                                                c_string_or_default(mode, "connectivity"),
                                                bandwidth,
                                                self != 0),
            out);
    });
}

int librosa_timelag_median_filter(const double *s,
                                  int64_t rows,
                                  int64_t columns,
                                  int size_rows,
                                  int size_columns,
                                  int pad,
                                  LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_matrix_input(s, rows, columns)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::segment::timelag_filter(
                make_matrix(s, rows, columns),
                [size_rows, size_columns](const librosa::ArrayXXr& lag) {
                    return librosa::decompose::median_filter_2d(lag, {size_rows, size_columns});
                },
                pad != 0),
            out);
    });
}

int librosa_laplacian_components(const double *s,
                                 int64_t rows,
                                 int64_t columns,
                                 int components,
                                 int median_filter_rows,
                                 LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_matrix_input(s, rows, columns)) {
            return LIBROSA_STATUS_ERROR;
        }
        if (rows != columns) {
            return fail_with("affinity matrix must be square");
        }
        if (components < 1 || components > columns) {
            return fail_with("components must be in [1, columns]");
        }

        librosa::ArrayXXr affinity = make_matrix(s, rows, columns);
        const Eigen::Index n = affinity.rows();
        librosa::ArrayXr degree = affinity.rowwise().sum();
        librosa::MatrixXr laplacian = librosa::MatrixXr::Zero(n, n);

        for (Eigen::Index i = 0; i < n; ++i) {
            if (degree(i) > 0) {
                laplacian(i, i) = 1.0;
            }
        }
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = 0; j < n; ++j) {
                if (affinity(i, j) != 0.0 && degree(i) > 0 && degree(j) > 0) {
                    laplacian(i, j) -= affinity(i, j) / std::sqrt(degree(i) * degree(j));
                }
            }
        }

        Eigen::SelfAdjointEigenSolver<librosa::MatrixXr> solver(laplacian);
        if (solver.info() != Eigen::Success) {
            return fail_with("failed to compute Laplacian eigenvectors");
        }

        librosa::ArrayXXr eigenvectors = solver.eigenvectors().array();
        if (median_filter_rows > 1) {
            eigenvectors = librosa::decompose::median_filter_2d(eigenvectors, {median_filter_rows, 1});
        }

        librosa::ArrayXXr result(n, components);
        for (Eigen::Index row = 0; row < n; ++row) {
            double norm = 0.0;
            for (int column = 0; column < components; ++column) {
                norm += eigenvectors(row, column) * eigenvectors(row, column);
            }
            norm = std::sqrt(std::max(norm, std::numeric_limits<double>::epsilon()));
            for (int column = 0; column < components; ++column) {
                result(row, column) = eigenvectors(row, column) / norm;
            }
        }

        return copy_matrix(result, out);
    });
}

int librosa_melspectrogram(const double *y,
                           int64_t count,
                           double sample_rate,
                           int n_fft,
                           int hop_length,
                           int n_mels,
                           double fmin,
                           int has_fmax,
                           double fmax,
                           int htk,
                           int norm_slaney,
                           LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::melspectrogram(make_array(y, count),
                                             sample_rate,
                                             n_fft,
                                             hop_length,
                                             std::nullopt,
                                             librosa::WindowType::Hann,
                                             true,
                                             librosa::PadMode::Constant,
                                             2.0,
                                             n_mels,
                                             fmin,
                                             optional_real(has_fmax, fmax),
                                             htk != 0,
                                             norm_slaney != 0),
            out);
    });
}

int librosa_mfcc(const double *y,
                 int64_t count,
                 double sample_rate,
                 int n_mfcc,
                 int n_fft,
                 int hop_length,
                 int n_mels,
                 double fmin,
                 int has_fmax,
                 double fmax,
                 int htk,
                 LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::mfcc(make_array(y, count),
                                   sample_rate,
                                   n_mfcc,
                                   2,
                                   true,
                                   0,
                                   n_fft,
                                   hop_length,
                                   n_mels,
                                   fmin,
                                   optional_real(has_fmax, fmax),
                                   htk != 0),
            out);
    });
}

int librosa_chroma_stft(const double *y,
                        int64_t count,
                        double sample_rate,
                        int n_fft,
                        int hop_length,
                        int n_chroma,
                        LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::chroma_stft(make_array(y, count),
                                          sample_rate,
                                          n_fft,
                                          hop_length,
                                          n_chroma),
            out);
    });
}

int librosa_chroma_stft_options(const double *y,
                                int64_t count,
                                double sample_rate,
                                int n_fft,
                                int hop_length,
                                int n_chroma,
                                int has_tuning,
                                double tuning,
                                double norm,
                                LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::chroma_stft(make_array(y, count),
                                          sample_rate,
                                          n_fft,
                                          hop_length,
                                          n_chroma,
                                          optional_real(has_tuning, tuning),
                                          norm),
            out);
    });
}

int librosa_dtw(const double *x,
                int64_t x_rows,
                int64_t x_columns,
                const double *y,
                int64_t y_rows,
                int64_t y_columns,
                const char *metric,
                int subseq,
                LibrosaDTWResult *out) {
    return run([&]() {
        if (!validate_out(out, "out")) {
            return LIBROSA_STATUS_ERROR;
        }
        out->cost = LibrosaMatrix{0, 0, nullptr};
        out->path = LibrosaIndexVector{0, nullptr};

        if (!validate_matrix_input(x, x_rows, x_columns) ||
            !validate_matrix_input(y, y_rows, y_columns)) {
            return LIBROSA_STATUS_ERROR;
        }
        if (x_rows != y_rows) {
            return fail_with("DTW feature matrices must have the same row count");
        }

        auto [cost, path] = librosa::sequence::dtw_backtrack(
            make_matrix(x, x_rows, x_columns),
            make_matrix(y, y_rows, y_columns),
            subseq != 0,
            c_string_or_default(metric, "euclidean"));

        int status = copy_matrix(cost, &out->cost);
        if (status != LIBROSA_STATUS_OK) {
            return status;
        }

        status = copy_path(path, &out->path);
        if (status != LIBROSA_STATUS_OK) {
            librosa_matrix_free(&out->cost);
            return status;
        }

        return LIBROSA_STATUS_OK;
    });
}

int librosa_spectral_centroid(const double *y,
                              int64_t count,
                              double sample_rate,
                              int n_fft,
                              int hop_length,
                              LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::spectral_centroid(make_array(y, count),
                                                sample_rate,
                                                n_fft,
                                                hop_length),
            out);
    });
}

int librosa_spectral_bandwidth(const double *y,
                               int64_t count,
                               double sample_rate,
                               int n_fft,
                               int hop_length,
                               double p,
                               int norm,
                               LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::spectral_bandwidth(make_array(y, count),
                                                 sample_rate,
                                                 n_fft,
                                                 hop_length,
                                                 librosa::WindowType::Hann,
                                                 true,
                                                 p,
                                                 norm != 0),
            out);
    });
}

int librosa_spectral_rolloff(const double *y,
                             int64_t count,
                             double sample_rate,
                             int n_fft,
                             int hop_length,
                             double roll_percent,
                             LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::spectral_rolloff(make_array(y, count),
                                               sample_rate,
                                               n_fft,
                                               hop_length,
                                               librosa::WindowType::Hann,
                                               true,
                                               roll_percent),
            out);
    });
}

int librosa_spectral_flatness(const double *y,
                              int64_t count,
                              int n_fft,
                              int hop_length,
                              LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::spectral_flatness(make_array(y, count),
                                                n_fft,
                                                hop_length),
            out);
    });
}

int librosa_spectral_contrast(const double *y,
                              int64_t count,
                              double sample_rate,
                              int n_fft,
                              int hop_length,
                              double fmin,
                              int n_bands,
                              double quantile,
                              int linear,
                              LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::spectral_contrast(make_array(y, count),
                                                sample_rate,
                                                n_fft,
                                                hop_length,
                                                librosa::WindowType::Hann,
                                                true,
                                                fmin,
                                                n_bands,
                                                quantile,
                                                linear != 0),
            out);
    });
}

int librosa_rms(const double *y,
                int64_t count,
                int frame_length,
                int hop_length,
                int center,
                LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::rms(make_array(y, count),
                                  frame_length,
                                  hop_length,
                                  center != 0),
            out);
    });
}

int librosa_zero_crossing_rate(const double *y,
                               int64_t count,
                               int frame_length,
                               int hop_length,
                               int center,
                               double threshold,
                               LibrosaMatrix *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_matrix(
            librosa::feature::zero_crossing_rate(make_array(y, count),
                                                 frame_length,
                                                 hop_length,
                                                 center != 0,
                                                 threshold),
            out);
    });
}

int librosa_onset_strength(const double *y,
                           int64_t count,
                           double sample_rate,
                           int n_fft,
                           int hop_length,
                           int lag,
                           int max_size,
                           int detrend,
                           int center,
                           LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(
            librosa::onset::onset_strength(make_array(y, count),
                                           sample_rate,
                                           n_fft,
                                           hop_length,
                                           lag,
                                           max_size,
                                           detrend != 0,
                                           center != 0),
            out);
    });
}

int librosa_onset_strength_spectrogram(const double *s,
                                       int64_t rows,
                                       int64_t columns,
                                       double sample_rate,
                                       int n_fft,
                                       int hop_length,
                                       int lag,
                                       int max_size,
                                       int detrend,
                                       int center,
                                       LibrosaVector *out) {
    return run([&]() {
        if (!validate_matrix_input(s, rows, columns)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(
            librosa::onset::onset_strength(make_matrix(s, rows, columns),
                                           sample_rate,
                                           n_fft,
                                           hop_length,
                                           lag,
                                           max_size,
                                           detrend != 0,
                                           center != 0),
            out);
    });
}

int librosa_onset_detect(const double *y,
                         int64_t count,
                         double sample_rate,
                         int hop_length,
                         int backtrack,
                         int normalize,
                         LibrosaIndexVector *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_indices(
            librosa::onset::onset_detect(make_array(y, count),
                                         sample_rate,
                                         hop_length,
                                         backtrack != 0,
                                         librosa::onset::OnsetUnits::Frames,
                                         normalize != 0),
            out);
    });
}

int librosa_onset_detect_envelope(const double *onset_envelope,
                                  int64_t count,
                                  double sample_rate,
                                  int hop_length,
                                  int backtrack,
                                  int normalize,
                                  LibrosaIndexVector *out) {
    return run([&]() {
        if (!validate_input(onset_envelope, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_indices(
            librosa::onset::onset_detect_envelope(make_array(onset_envelope, count),
                                                  sample_rate,
                                                  hop_length,
                                                  backtrack != 0,
                                                  librosa::onset::OnsetUnits::Frames,
                                                  normalize != 0),
            out);
    });
}

int librosa_tempo(const double *onset_envelope,
                  int64_t count,
                  double sample_rate,
                  int hop_length,
                  double start_bpm,
                  double std_bpm,
                  double ac_size,
                  int has_max_tempo,
                  double max_tempo,
                  double *out_tempo) {
    return run([&]() {
        if (!validate_input(onset_envelope, count) ||
            !validate_out(out_tempo, "out_tempo")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_tempo = librosa::beat::tempo(make_array(onset_envelope, count),
                                          sample_rate,
                                          hop_length,
                                          start_bpm,
                                          std_bpm,
                                          ac_size,
                                          optional_real(has_max_tempo, max_tempo),
                                          true);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_tempo_audio(const double *y,
                        int64_t count,
                        double sample_rate,
                        int hop_length,
                        double start_bpm,
                        double std_bpm,
                        double ac_size,
                        int has_max_tempo,
                        double max_tempo,
                        double *out_tempo) {
    return run([&]() {
        if (!validate_input(y, count) || !validate_out(out_tempo, "out_tempo")) {
            return LIBROSA_STATUS_ERROR;
        }
        *out_tempo = librosa::beat::tempo_audio(make_array(y, count),
                                                sample_rate,
                                                hop_length,
                                                start_bpm,
                                                std_bpm,
                                                ac_size,
                                                optional_real(has_max_tempo, max_tempo),
                                                true);
        return LIBROSA_STATUS_OK;
    });
}

int librosa_beat_track(const double *onset_envelope,
                       int64_t count,
                       double sample_rate,
                       int hop_length,
                       double start_bpm,
                       double tightness,
                       int trim,
                       int has_bpm,
                       double bpm,
                       LibrosaBeatTrackResult *out) {
    return run([&]() {
        if (!validate_input(onset_envelope, count) ||
            !validate_out(out, "out")) {
            return LIBROSA_STATUS_ERROR;
        }

        auto result = librosa::beat::beat_track(make_array(onset_envelope, count),
                                                sample_rate,
                                                hop_length,
                                                start_bpm,
                                                tightness,
                                                trim != 0,
                                                optional_real(has_bpm, bpm),
                                                librosa::beat::BeatUnits::Frames);
        out->tempo = result.first;
        out->beats = LibrosaIndexVector{0, nullptr};
        return copy_indices(result.second, &out->beats);
    });
}

int librosa_beat_track_audio(const double *y,
                             int64_t count,
                             double sample_rate,
                             int hop_length,
                             double start_bpm,
                             double tightness,
                             int trim,
                             int has_bpm,
                             double bpm,
                             LibrosaBeatTrackResult *out) {
    return run([&]() {
        if (!validate_input(y, count) ||
            !validate_out(out, "out")) {
            return LIBROSA_STATUS_ERROR;
        }

        auto result = librosa::beat::beat_track_audio(make_array(y, count),
                                                      sample_rate,
                                                      hop_length,
                                                      start_bpm,
                                                      tightness,
                                                      trim != 0,
                                                      optional_real(has_bpm, bpm),
                                                      librosa::beat::BeatUnits::Frames);
        out->tempo = result.first;
        out->beats = LibrosaIndexVector{0, nullptr};
        return copy_indices(result.second, &out->beats);
    });
}

int librosa_time_stretch(const double *y,
                         int64_t count,
                         double rate,
                         int n_fft,
                         int hop_length,
                         LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(
            librosa::effects::time_stretch(make_array(y, count),
                                           rate,
                                           n_fft,
                                           hop_length),
            out);
    });
}

int librosa_pitch_shift(const double *y,
                        int64_t count,
                        double sample_rate,
                        double steps,
                        int bins_per_octave,
                        const char *res_type,
                        int n_fft,
                        int hop_length,
                        LibrosaVector *out) {
    return run([&]() {
        if (!validate_input(y, count)) {
            return LIBROSA_STATUS_ERROR;
        }
        return copy_vector(
            librosa::effects::pitch_shift(make_array(y, count),
                                          sample_rate,
                                          steps,
                                          bins_per_octave,
                                          c_string_or_default(res_type, "kaiser_hq"),
                                          n_fft,
                                          hop_length),
            out);
    });
}

int librosa_trim(const double *y,
                 int64_t count,
                 double top_db,
                 int frame_length,
                 int hop_length,
                 LibrosaTrimResult *out) {
    return run([&]() {
        if (!validate_input(y, count) || !validate_out(out, "out")) {
            return LIBROSA_STATUS_ERROR;
        }

        auto result = librosa::effects::trim(make_array(y, count),
                                             top_db,
                                             -1,
                                             frame_length,
                                             hop_length);
        out->audio = LibrosaVector{0, nullptr};
        out->start = static_cast<int64_t>(result.second.first);
        out->end = static_cast<int64_t>(result.second.second);
        return copy_vector(result.first, &out->audio);
    });
}

} // extern "C"
