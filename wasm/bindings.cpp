#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <librosa/beat.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/core/constantq.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/harmonic.hpp>
#include <librosa/core/intervals.hpp>
#include <librosa/core/notation.hpp>
#include <librosa/core/pitch.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/decompose.hpp>
#include <librosa/effects.hpp>
#include <librosa/feature/inverse.hpp>
#include <librosa/feature/rhythm.hpp>
#include <librosa/feature/spectral.hpp>
#include <librosa/feature/utils.hpp>
#include <librosa/filters.hpp>
#include <librosa/onset.hpp>
#include <librosa/segment.hpp>
#include <librosa/sequence.hpp>
#include <librosa/util/exceptions.hpp>
#include <librosa/util/utils.hpp>

#if __has_include(<pffft/pffft_double.h>)
#include <pffft/pffft_double.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using emscripten::allow_raw_pointers;
using emscripten::function;
using emscripten::typed_memory_view;
using emscripten::val;

namespace {

using librosa::AggregateFunc;
using librosa::ArrayXc;
using librosa::ArrayXr;
using librosa::ArrayXXc;
using librosa::ArrayXXr;
using librosa::Complex;
using librosa::MatrixXr;
using librosa::PadMode;
using librosa::Real;
using librosa::WindowType;

bool is_nullish(const val& v) {
    return v.isUndefined() || v.isNull();
}

val prop(const val& object, const char* key) {
    if (is_nullish(object)) {
        return val::undefined();
    }
    return object[key];
}

std::string type_of(const val& v) {
    return v.typeOf().as<std::string>();
}

bool is_number(const val& v) {
    return type_of(v) == "number";
}

int int_prop(const val& object, const char* key, int default_value) {
    val v = prop(object, key);
    return v.isUndefined() ? default_value : v.as<int>();
}

long long index_prop(const val& object, const char* key, long long default_value) {
    val v = prop(object, key);
    return v.isUndefined() ? default_value : v.as<long long>();
}

Real real_prop(const val& object, const char* key, Real default_value) {
    val v = prop(object, key);
    return v.isUndefined() ? default_value : v.as<Real>();
}

bool bool_prop(const val& object, const char* key, bool default_value) {
    val v = prop(object, key);
    return v.isUndefined() ? default_value : v.as<bool>();
}

std::string string_prop(const val& object, const char* key, const std::string& default_value) {
    val v = prop(object, key);
    return v.isUndefined() ? default_value : v.as<std::string>();
}

std::optional<int> opt_int_prop(const val& object, const char* key,
                                std::optional<int> default_value = std::nullopt) {
    val v = prop(object, key);
    if (v.isUndefined()) return default_value;
    if (v.isNull()) return std::nullopt;
    return v.as<int>();
}

std::optional<long long> opt_index_prop(const val& object, const char* key,
                                        std::optional<long long> default_value = std::nullopt) {
    val v = prop(object, key);
    if (v.isUndefined()) return default_value;
    if (v.isNull()) return std::nullopt;
    return v.as<long long>();
}

std::optional<Real> opt_real_prop(const val& object, const char* key,
                                  std::optional<Real> default_value = std::nullopt) {
    val v = prop(object, key);
    if (v.isUndefined()) return default_value;
    if (v.isNull()) return std::nullopt;
    return v.as<Real>();
}

std::optional<bool> opt_bool_prop(const val& object, const char* key,
                                  std::optional<bool> default_value = std::nullopt) {
    val v = prop(object, key);
    if (v.isUndefined()) return default_value;
    if (v.isNull()) return std::nullopt;
    return v.as<bool>();
}

WindowType window_from_string(const std::string& name) {
    if (name == "hann" || name == "hanning") return WindowType::Hann;
    if (name == "hamming") return WindowType::Hamming;
    if (name == "blackman") return WindowType::Blackman;
    if (name == "bartlett") return WindowType::Bartlett;
    if (name == "rectangular" || name == "rect" || name == "boxcar") return WindowType::Rectangular;
    if (name == "kaiser") return WindowType::Kaiser;
    if (name == "gaussian") return WindowType::Gaussian;
    if (name == "tukey") return WindowType::Tukey;
    if (name == "triangle" || name == "triang") return WindowType::Triangle;
    throw std::invalid_argument("Unsupported window type: " + name);
}

WindowType window_prop(const val& object, const char* key, WindowType default_value) {
    val v = prop(object, key);
    if (v.isUndefined() || v.isNull()) return default_value;
    return window_from_string(v.as<std::string>());
}

PadMode pad_mode_from_string(const std::string& name) {
    if (name == "constant") return PadMode::Constant;
    if (name == "edge") return PadMode::Edge;
    if (name == "reflect") return PadMode::Reflect;
    if (name == "symmetric") return PadMode::Symmetric;
    if (name == "wrap") return PadMode::Wrap;
    if (name == "linear_ramp" || name == "linear-ramp") return PadMode::Linear_ramp;
    throw std::invalid_argument("Unsupported pad mode: " + name);
}

PadMode pad_mode_prop(const val& object, const char* key, PadMode default_value) {
    val v = prop(object, key);
    if (v.isUndefined() || v.isNull()) return default_value;
    return pad_mode_from_string(v.as<std::string>());
}

librosa::WeightType weight_type_from_string(const std::string& name) {
    if (name == "A" || name == "a") return librosa::WeightType::A;
    if (name == "B" || name == "b") return librosa::WeightType::B;
    if (name == "C" || name == "c") return librosa::WeightType::C;
    if (name == "D" || name == "d") return librosa::WeightType::D;
    if (name == "Z" || name == "z") return librosa::WeightType::Z;
    throw std::invalid_argument("Unsupported weighting type: " + name);
}

librosa::filters::MelNorm mel_norm_from_string(const std::string& name) {
    if (name == "none") return librosa::filters::MelNorm::None;
    if (name == "slaney") return librosa::filters::MelNorm::Slaney;
    if (name == "l1" || name == "1") return librosa::filters::MelNorm::L1;
    if (name == "l2" || name == "2") return librosa::filters::MelNorm::L2;
    if (name == "inf" || name == "infinity") return librosa::filters::MelNorm::Inf;
    throw std::invalid_argument("Unsupported mel normalization: " + name);
}

AggregateFunc aggregate_from_string(const std::string& name) {
    if (name == "mean") return AggregateFunc::Mean;
    if (name == "median") return AggregateFunc::Median;
    if (name == "min") return AggregateFunc::Min;
    if (name == "max") return AggregateFunc::Max;
    throw std::invalid_argument("Unsupported aggregate function: " + name);
}

librosa::beat::BeatUnits beat_units_from_string(const std::string& name) {
    if (name == "frames") return librosa::beat::BeatUnits::Frames;
    if (name == "samples") return librosa::beat::BeatUnits::Samples;
    if (name == "time") return librosa::beat::BeatUnits::Time;
    throw std::invalid_argument("Unsupported beat units: " + name);
}

librosa::onset::OnsetUnits onset_units_from_string(const std::string& name) {
    if (name == "frames") return librosa::onset::OnsetUnits::Frames;
    if (name == "samples") return librosa::onset::OnsetUnits::Samples;
    if (name == "time") return librosa::onset::OnsetUnits::Time;
    throw std::invalid_argument("Unsupported onset units: " + name);
}

ArrayXr vector_from_js(const val& input) {
    val data = prop(input, "data");
    const val& values = data.isUndefined() ? input : data;
    const unsigned length = values["length"].as<unsigned>();
    ArrayXr out(length);
    for (unsigned i = 0; i < length; ++i) {
        out(static_cast<Eigen::Index>(i)) = values[i].as<Real>();
    }
    return out;
}

std::vector<Real> real_vector_from_js(const val& input) {
    ArrayXr values = vector_from_js(input);
    return std::vector<Real>(values.data(), values.data() + values.size());
}

std::vector<int> int_vector_from_js(const val& input) {
    const unsigned length = input["length"].as<unsigned>();
    std::vector<int> out(length);
    for (unsigned i = 0; i < length; ++i) {
        out[i] = input[i].as<int>();
    }
    return out;
}

std::vector<Eigen::Index> index_vector_from_js(const val& input) {
    const unsigned length = input["length"].as<unsigned>();
    std::vector<Eigen::Index> out(length);
    for (unsigned i = 0; i < length; ++i) {
        out[i] = static_cast<Eigen::Index>(input[i].as<long long>());
    }
    return out;
}

std::vector<std::string> string_vector_from_js(const val& input) {
    const unsigned length = input["length"].as<unsigned>();
    std::vector<std::string> out(length);
    for (unsigned i = 0; i < length; ++i) {
        out[i] = input[i].as<std::string>();
    }
    return out;
}

std::vector<std::pair<Eigen::Index, Eigen::Index>> intervals_from_js(const val& input) {
    std::vector<std::pair<Eigen::Index, Eigen::Index>> out;
    val rows_v = prop(input, "rows");
    if (!rows_v.isUndefined()) {
        ArrayXXr matrix(input["rows"].as<int>(), input["cols"].as<int>());
        val data = input["data"];
        if (matrix.cols() != 2) {
            throw std::invalid_argument("interval matrix must have 2 columns");
        }
        for (Eigen::Index r = 0; r < matrix.rows(); ++r) {
            out.emplace_back(static_cast<Eigen::Index>(data[static_cast<int>(r * 2)].as<long long>()),
                             static_cast<Eigen::Index>(data[static_cast<int>(r * 2 + 1)].as<long long>()));
        }
        return out;
    }

    const unsigned length = input["length"].as<unsigned>();
    out.reserve(length);
    for (unsigned i = 0; i < length; ++i) {
        val pair = input[i];
        out.emplace_back(static_cast<Eigen::Index>(pair[0].as<long long>()),
                         static_cast<Eigen::Index>(pair[1].as<long long>()));
    }
    return out;
}

ArrayXXr matrix_from_js(const val& input) {
    const int rows = input["rows"].as<int>();
    const int cols = input["cols"].as<int>();
    val data = input["data"];
    ArrayXXr out(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out(r, c) = data[r * cols + c].as<Real>();
        }
    }
    return out;
}

MatrixXr dense_matrix_from_js(const val& input) {
    return matrix_from_js(input).matrix();
}

ArrayXXc complex_matrix_from_js(const val& input) {
    const int rows = input["rows"].as<int>();
    const int cols = input["cols"].as<int>();
    val data = input["data"];
    ArrayXXc out(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int idx = 2 * (r * cols + c);
            out(r, c) = Complex(data[idx].as<Real>(), data[idx + 1].as<Real>());
        }
    }
    return out;
}

ArrayXc complex_vector_from_js(const val& input) {
    val data = prop(input, "data");
    const val& values = data.isUndefined() ? input : data;
    const unsigned pairs = values["length"].as<unsigned>() / 2;
    ArrayXc out(pairs);
    for (unsigned i = 0; i < pairs; ++i) {
        out(static_cast<Eigen::Index>(i)) =
            Complex(values[2 * i].as<Real>(), values[2 * i + 1].as<Real>());
    }
    return out;
}

val float64_array_from_buffer(const Real* data, size_t length) {
    val out = val::global("Float64Array").new_(static_cast<unsigned>(length));
    if (length > 0) {
        out.call<void>("set", val(typed_memory_view(length, data)));
    }
    return out;
}

val vector_to_js(const ArrayXr& input) {
    return float64_array_from_buffer(input.data(), static_cast<size_t>(input.size()));
}

val matrix_to_js(const ArrayXXr& input) {
    val out = val::object();
    out.set("rows", static_cast<int>(input.rows()));
    out.set("cols", static_cast<int>(input.cols()));
    out.set("data", float64_array_from_buffer(input.data(), static_cast<size_t>(input.size())));
    return out;
}

val complex_vector_to_js(const ArrayXc& input) {
    std::vector<Real> interleaved(static_cast<size_t>(input.size()) * 2);
    for (Eigen::Index i = 0; i < input.size(); ++i) {
        interleaved[static_cast<size_t>(2 * i)] = input(i).real();
        interleaved[static_cast<size_t>(2 * i + 1)] = input(i).imag();
    }
    return float64_array_from_buffer(interleaved.data(), interleaved.size());
}

val complex_matrix_to_js(const ArrayXXc& input) {
    std::vector<Real> interleaved(static_cast<size_t>(input.size()) * 2);
    for (Eigen::Index r = 0; r < input.rows(); ++r) {
        for (Eigen::Index c = 0; c < input.cols(); ++c) {
            const size_t idx = static_cast<size_t>(2 * (r * input.cols() + c));
            interleaved[idx] = input(r, c).real();
            interleaved[idx + 1] = input(r, c).imag();
        }
    }
    val out = val::object();
    out.set("rows", static_cast<int>(input.rows()));
    out.set("cols", static_cast<int>(input.cols()));
    out.set("data", float64_array_from_buffer(interleaved.data(), interleaved.size()));
    return out;
}

val bool_vector_to_js(const Eigen::Array<bool, Eigen::Dynamic, 1>& input) {
    std::vector<uint8_t> bytes(static_cast<size_t>(input.size()));
    for (Eigen::Index i = 0; i < input.size(); ++i) {
        bytes[static_cast<size_t>(i)] = input(i) ? 1 : 0;
    }
    val out = val::global("Uint8Array").new_(static_cast<unsigned>(bytes.size()));
    if (!bytes.empty()) {
        out.call<void>("set", val(typed_memory_view(bytes.size(), bytes.data())));
    }
    return out;
}

val bool_matrix_to_js(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& input) {
    std::vector<uint8_t> bytes(static_cast<size_t>(input.size()));
    for (Eigen::Index r = 0; r < input.rows(); ++r) {
        for (Eigen::Index c = 0; c < input.cols(); ++c) {
            bytes[static_cast<size_t>(r * input.cols() + c)] = input(r, c) ? 1 : 0;
        }
    }
    val data = val::global("Uint8Array").new_(static_cast<unsigned>(bytes.size()));
    if (!bytes.empty()) {
        data.call<void>("set", val(typed_memory_view(bytes.size(), bytes.data())));
    }
    val out = val::object();
    out.set("rows", static_cast<int>(input.rows()));
    out.set("cols", static_cast<int>(input.cols()));
    out.set("data", data);
    return out;
}

val int_vector_to_js(const std::vector<int>& input) {
    val out = val::global("Int32Array").new_(static_cast<unsigned>(input.size()));
    if (!input.empty()) {
        out.call<void>("set", val(typed_memory_view(input.size(), input.data())));
    }
    return out;
}

val index_vector_to_js(const std::vector<Eigen::Index>& input) {
    std::vector<Real> values(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        values[i] = static_cast<Real>(input[i]);
    }
    return float64_array_from_buffer(values.data(), values.size());
}

val pair_index_vector_to_js(const std::vector<std::pair<Eigen::Index, Eigen::Index>>& input) {
    std::vector<Real> values(input.size() * 2);
    for (size_t i = 0; i < input.size(); ++i) {
        values[2 * i] = static_cast<Real>(input[i].first);
        values[2 * i + 1] = static_cast<Real>(input[i].second);
    }
    val out = val::object();
    out.set("rows", static_cast<int>(input.size()));
    out.set("cols", 2);
    out.set("data", float64_array_from_buffer(values.data(), values.size()));
    return out;
}

val path_to_js(const std::vector<std::pair<int, int>>& input) {
    std::vector<Real> values(input.size() * 2);
    for (size_t i = 0; i < input.size(); ++i) {
        values[2 * i] = static_cast<Real>(input[i].first);
        values[2 * i + 1] = static_cast<Real>(input[i].second);
    }
    val out = val::object();
    out.set("rows", static_cast<int>(input.size()));
    out.set("cols", 2);
    out.set("data", float64_array_from_buffer(values.data(), values.size()));
    return out;
}

val string_vector_to_js(const std::vector<std::string>& input) {
    val out = val::array();
    for (size_t i = 0; i < input.size(); ++i) {
        out.set(static_cast<unsigned>(i), input[i]);
    }
    return out;
}

val int_std_vector_to_js(const std::vector<int>& input) {
    return int_vector_to_js(input);
}

val map_string_int_to_js(const std::map<std::string, int>& input) {
    val out = val::object();
    for (const auto& [key, value] : input) {
        out.set(key, value);
    }
    return out;
}

template<typename Fn>
val protect(Fn&& fn) {
    try {
        return fn();
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}

std::optional<unsigned int> opt_uint_from_optional_int(std::optional<int> value) {
    if (!value) return std::nullopt;
    if (*value < 0) throw std::invalid_argument("unsigned option cannot be negative");
    return static_cast<unsigned int>(*value);
}

val build_info() {
    val out = val::object();
    out.set("fftBackend", "pffft");
    out.set("wasmSimd", static_cast<bool>(
#ifdef __wasm_simd128__
        true
#else
        false
#endif
    ));
#if __has_include(<pffft/pffft_double.h>)
    out.set("pffftSimdSize", pffftd_simd_size());
    out.set("pffftSimdArch", std::string(pffftd_simd_arch()));
#else
    out.set("pffftSimdSize", 0);
    out.set("pffftSimdArch", "unavailable");
#endif
    return out;
}

// ---------------------------------------------------------------------------
// Core conversions
// ---------------------------------------------------------------------------

val frames_to_samples_js(val frames, val options) {
    return protect([&] {
        int hop = int_prop(options, "hopLength", 512);
        auto n_fft = opt_int_prop(options, "nFft");
        if (is_number(frames)) {
            return val(static_cast<Real>(librosa::frames_to_samples(
                static_cast<Eigen::Index>(frames.as<long long>()), hop, n_fft)));
        }
        return vector_to_js(librosa::frames_to_samples(vector_from_js(frames), hop, n_fft));
    });
}

val samples_to_frames_js(val samples, val options) {
    return protect([&] {
        int hop = int_prop(options, "hopLength", 512);
        auto n_fft = opt_int_prop(options, "nFft");
        if (is_number(samples)) {
            return val(static_cast<Real>(librosa::samples_to_frames(
                static_cast<Eigen::Index>(samples.as<long long>()), hop, n_fft)));
        }
        return vector_to_js(librosa::samples_to_frames(vector_from_js(samples), hop, n_fft));
    });
}

val frames_to_time_js(val frames, val options) {
    return protect([&] {
        Real sr = real_prop(options, "sr", 22050);
        int hop = int_prop(options, "hopLength", 512);
        auto n_fft = opt_int_prop(options, "nFft");
        if (is_number(frames)) {
            return val(librosa::frames_to_time(static_cast<Eigen::Index>(frames.as<long long>()), sr, hop, n_fft));
        }
        return vector_to_js(librosa::frames_to_time(vector_from_js(frames), sr, hop, n_fft));
    });
}

val time_to_frames_js(val times, val options) {
    return protect([&] {
        Real sr = real_prop(options, "sr", 22050);
        int hop = int_prop(options, "hopLength", 512);
        auto n_fft = opt_int_prop(options, "nFft");
        if (is_number(times)) {
            return val(static_cast<Real>(librosa::time_to_frames(times.as<Real>(), sr, hop, n_fft)));
        }
        return vector_to_js(librosa::time_to_frames(vector_from_js(times), sr, hop, n_fft));
    });
}

val time_to_samples_js(val times, val options) {
    return protect([&] {
        Real sr = real_prop(options, "sr", 22050);
        if (is_number(times)) {
            return val(static_cast<Real>(librosa::time_to_samples(times.as<Real>(), sr)));
        }
        return vector_to_js(librosa::time_to_samples(vector_from_js(times), sr));
    });
}

val samples_to_time_js(val samples, val options) {
    return protect([&] {
        Real sr = real_prop(options, "sr", 22050);
        if (is_number(samples)) {
            return val(librosa::samples_to_time(static_cast<Eigen::Index>(samples.as<long long>()), sr));
        }
        return vector_to_js(librosa::samples_to_time(vector_from_js(samples), sr));
    });
}

val midi_to_hz_js(val midi) {
    return protect([&] {
        if (is_number(midi)) return val(librosa::midi_to_hz(midi.as<Real>()));
        return vector_to_js(librosa::midi_to_hz(vector_from_js(midi)));
    });
}

val hz_to_midi_js(val frequency) {
    return protect([&] {
        if (is_number(frequency)) return val(librosa::hz_to_midi(frequency.as<Real>()));
        return vector_to_js(librosa::hz_to_midi(vector_from_js(frequency)));
    });
}

val note_to_midi_js(val note, val options) {
    return protect([&] {
        bool round_midi = bool_prop(options, "roundMidi", true);
        if (type_of(note) == "string") return val(librosa::note_to_midi(note.as<std::string>(), round_midi));
        return vector_to_js(librosa::note_to_midi(string_vector_from_js(note), round_midi));
    });
}

val note_to_hz_js(val note) {
    return protect([&] {
        if (type_of(note) == "string") return val(librosa::note_to_hz(note.as<std::string>()));
        return vector_to_js(librosa::note_to_hz(string_vector_from_js(note)));
    });
}

val midi_to_note_js(val midi, val options) {
    return protect([&] {
        bool octave = bool_prop(options, "octave", true);
        bool cents = bool_prop(options, "cents", false);
        bool unicode = bool_prop(options, "unicode", true);
        if (is_number(midi)) return val(librosa::midi_to_note(midi.as<Real>(), octave, cents, unicode));
        return string_vector_to_js(librosa::midi_to_note(vector_from_js(midi), octave, cents, unicode));
    });
}

val hz_to_note_js(val frequency, val options) {
    return protect([&] {
        bool octave = bool_prop(options, "octave", true);
        bool cents = bool_prop(options, "cents", false);
        bool unicode = bool_prop(options, "unicode", true);
        if (is_number(frequency)) return val(librosa::hz_to_note(frequency.as<Real>(), octave, cents, unicode));
        return string_vector_to_js(librosa::hz_to_note(vector_from_js(frequency), octave, cents, unicode));
    });
}

val hz_to_mel_js(val frequency, val options) {
    return protect([&] {
        bool htk = bool_prop(options, "htk", false);
        if (is_number(frequency)) return val(librosa::hz_to_mel(frequency.as<Real>(), htk));
        return vector_to_js(librosa::hz_to_mel(vector_from_js(frequency), htk));
    });
}

val mel_to_hz_js(val mel, val options) {
    return protect([&] {
        bool htk = bool_prop(options, "htk", false);
        if (is_number(mel)) return val(librosa::mel_to_hz(mel.as<Real>(), htk));
        return vector_to_js(librosa::mel_to_hz(vector_from_js(mel), htk));
    });
}

val hz_to_octs_js(val frequency, val options) {
    return protect([&] {
        Real tuning = real_prop(options, "tuning", 0);
        int bins = int_prop(options, "binsPerOctave", 12);
        if (is_number(frequency)) return val(librosa::hz_to_octs(frequency.as<Real>(), tuning, bins));
        return vector_to_js(librosa::hz_to_octs(vector_from_js(frequency), tuning, bins));
    });
}

val octs_to_hz_js(val octs, val options) {
    return protect([&] {
        Real tuning = real_prop(options, "tuning", 0);
        int bins = int_prop(options, "binsPerOctave", 12);
        if (is_number(octs)) return val(librosa::octs_to_hz(octs.as<Real>(), tuning, bins));
        return vector_to_js(librosa::octs_to_hz(vector_from_js(octs), tuning, bins));
    });
}

val a4_to_tuning_js(val a4, val options) {
    return protect([&] {
        int bins = int_prop(options, "binsPerOctave", 12);
        if (is_number(a4)) return val(librosa::A4_to_tuning(a4.as<Real>(), bins));
        return vector_to_js(librosa::A4_to_tuning(vector_from_js(a4), bins));
    });
}

val tuning_to_a4_js(val tuning, val options) {
    return protect([&] {
        int bins = int_prop(options, "binsPerOctave", 12);
        if (is_number(tuning)) return val(librosa::tuning_to_A4(tuning.as<Real>(), bins));
        return vector_to_js(librosa::tuning_to_A4(vector_from_js(tuning), bins));
    });
}

val fft_frequencies_js(val options) {
    return protect([&] {
        return vector_to_js(librosa::fft_frequencies(real_prop(options, "sr", 22050), int_prop(options, "nFft", 2048)));
    });
}

val cqt_frequencies_js(val options) {
    return protect([&] {
        return vector_to_js(librosa::cqt_frequencies(int_prop(options, "nBins", 84),
                                                     real_prop(options, "fmin", 32.70319566257483),
                                                     int_prop(options, "binsPerOctave", 12),
                                                     real_prop(options, "tuning", 0)));
    });
}

val mel_frequencies_js(val options) {
    return protect([&] {
        return vector_to_js(librosa::mel_frequencies(int_prop(options, "nMels", 128),
                                                     real_prop(options, "fmin", 0),
                                                     real_prop(options, "fmax", 11025),
                                                     bool_prop(options, "htk", false)));
    });
}

val tempo_frequencies_js(val options) {
    return protect([&] {
        return vector_to_js(librosa::tempo_frequencies(int_prop(options, "nBins", 384),
                                                       int_prop(options, "hopLength", 512),
                                                       real_prop(options, "sr", 22050)));
    });
}

val fourier_tempo_frequencies_js(val options) {
    return protect([&] {
        return vector_to_js(librosa::fourier_tempo_frequencies(real_prop(options, "sr", 22050),
                                                               int_prop(options, "winLength", 384),
                                                               int_prop(options, "hopLength", 512)));
    });
}

val weighting_js(val frequency, val options) {
    return protect([&] {
        auto min_db = opt_real_prop(options, "minDb", -80.0);
        std::string kind = string_prop(options, "kind", "A");
        auto apply_scalar = [&](Real x) -> Real {
            switch (weight_type_from_string(kind)) {
            case librosa::WeightType::A: return librosa::A_weighting(x, min_db);
            case librosa::WeightType::B: return librosa::B_weighting(x, min_db);
            case librosa::WeightType::C: return librosa::C_weighting(x, min_db);
            case librosa::WeightType::D: return librosa::D_weighting(x, min_db);
            case librosa::WeightType::Z: return librosa::Z_weighting(x, min_db);
            }
            return 0;
        };
        if (is_number(frequency)) return val(apply_scalar(frequency.as<Real>()));
        ArrayXr f = vector_from_js(frequency);
        switch (weight_type_from_string(kind)) {
        case librosa::WeightType::A: return vector_to_js(librosa::A_weighting(f, min_db));
        case librosa::WeightType::B: return vector_to_js(librosa::B_weighting(f, min_db));
        case librosa::WeightType::C: return vector_to_js(librosa::C_weighting(f, min_db));
        case librosa::WeightType::D: return vector_to_js(librosa::D_weighting(f, min_db));
        case librosa::WeightType::Z: return vector_to_js(librosa::Z_weighting(f, min_db));
        }
        return val::undefined();
    });
}

// ---------------------------------------------------------------------------
// Audio utilities and synthesis
// ---------------------------------------------------------------------------

val to_mono_js(val y) {
    return protect([&] {
        if (!prop(y, "rows").isUndefined()) {
            return vector_to_js(librosa::to_mono(matrix_from_js(y)));
        }
        return vector_to_js(librosa::to_mono(vector_from_js(y)));
    });
}

val resample_js(val y, val options) {
    return protect([&] {
        return vector_to_js(librosa::resample(vector_from_js(y),
                                              real_prop(options, "origSr", 22050),
                                              real_prop(options, "targetSr", 22050),
                                              string_prop(options, "resType", "kaiser_hq"),
                                              bool_prop(options, "fix", true),
                                              bool_prop(options, "scale", false)));
    });
}

val autocorrelate_js(val y, val options) {
    return protect([&] {
        auto max_size = opt_int_prop(options, "maxSize");
        if (!prop(y, "rows").isUndefined()) {
            return matrix_to_js(librosa::autocorrelate(matrix_from_js(y), max_size, int_prop(options, "axis", -1)));
        }
        return vector_to_js(librosa::autocorrelate(vector_from_js(y), max_size));
    });
}

val lpc_js(val y, val options) {
    return protect([&] {
        int order = int_prop(options, "order", 2);
        if (!prop(y, "rows").isUndefined()) {
            return matrix_to_js(librosa::lpc(matrix_from_js(y), order, int_prop(options, "axis", -1)));
        }
        return vector_to_js(librosa::lpc(vector_from_js(y), order));
    });
}

val zero_crossings_js(val y, val options) {
    return protect([&] {
        Real threshold = real_prop(options, "threshold", 1e-10);
        auto ref = opt_real_prop(options, "refMagnitude");
        bool pad = bool_prop(options, "pad", true);
        bool zero_pos = bool_prop(options, "zeroPos", true);
        if (!prop(y, "rows").isUndefined()) {
            return bool_matrix_to_js(librosa::zero_crossings(matrix_from_js(y), threshold, ref, pad, zero_pos,
                                                            int_prop(options, "axis", -1)));
        }
        return bool_vector_to_js(librosa::zero_crossings(vector_from_js(y), threshold, ref, pad, zero_pos));
    });
}

val clicks_js(val times, val options) {
    return protect([&] {
        return vector_to_js(librosa::clicks(vector_from_js(times),
                                            real_prop(options, "sr", 22050),
                                            real_prop(options, "clickFreq", 1000),
                                            real_prop(options, "clickDuration", 0.1),
                                            opt_int_prop(options, "length")));
    });
}

val clicks_frames_js(val frames, val options) {
    return protect([&] {
        return vector_to_js(librosa::clicks_frames(index_vector_from_js(frames),
                                                   real_prop(options, "sr", 22050),
                                                   int_prop(options, "hopLength", 512),
                                                   real_prop(options, "clickFreq", 1000),
                                                   real_prop(options, "clickDuration", 0.1),
                                                   opt_int_prop(options, "length")));
    });
}

val tone_js(val frequency, val options) {
    return protect([&] {
        return vector_to_js(librosa::tone(frequency.as<Real>(),
                                          real_prop(options, "sr", 22050),
                                          opt_int_prop(options, "length"),
                                          opt_real_prop(options, "duration"),
                                          opt_real_prop(options, "phi")));
    });
}

val chirp_js(val options) {
    return protect([&] {
        return vector_to_js(librosa::chirp(real_prop(options, "fmin", 32.7),
                                           real_prop(options, "fmax", 880),
                                           real_prop(options, "sr", 22050),
                                           opt_int_prop(options, "length"),
                                           opt_real_prop(options, "duration"),
                                           bool_prop(options, "linear", false),
                                           opt_real_prop(options, "phi")));
    });
}

val mu_compress_js(val x, val options) {
    return protect([&] {
        return vector_to_js(librosa::mu_compress(vector_from_js(x), real_prop(options, "mu", 255),
                                                 bool_prop(options, "quantize", true)));
    });
}

val mu_expand_js(val x, val options) {
    return protect([&] {
        return vector_to_js(librosa::mu_expand(vector_from_js(x), real_prop(options, "mu", 255),
                                               bool_prop(options, "quantize", true)));
    });
}

val duration_js(val input, val options) {
    return protect([&] {
        Real sr = real_prop(options, "sr", 22050);
        if (!prop(input, "rows").isUndefined()) {
            return val(librosa::get_duration(matrix_from_js(input), sr,
                                             int_prop(options, "hopLength", 512),
                                             int_prop(options, "nFft", 2048),
                                             bool_prop(options, "center", true)));
        }
        return val(librosa::get_duration(vector_from_js(input), sr));
    });
}

// ---------------------------------------------------------------------------
// Spectrum
// ---------------------------------------------------------------------------

val stft_js(val y, val options) {
    return protect([&] {
        return complex_matrix_to_js(librosa::stft(vector_from_js(y),
                                                  int_prop(options, "nFft", 2048),
                                                  opt_int_prop(options, "hopLength"),
                                                  opt_int_prop(options, "winLength"),
                                                  window_prop(options, "window", WindowType::Hann),
                                                  bool_prop(options, "center", true),
                                                  pad_mode_prop(options, "padMode", PadMode::Constant)));
    });
}

val istft_js(val d, val options) {
    return protect([&] {
        return vector_to_js(librosa::istft(complex_matrix_from_js(d),
                                           opt_int_prop(options, "hopLength"),
                                           opt_int_prop(options, "winLength"),
                                           opt_int_prop(options, "nFft"),
                                           window_prop(options, "window", WindowType::Hann),
                                           bool_prop(options, "center", true),
                                           opt_int_prop(options, "length")));
    });
}

val magphase_js(val d, val options) {
    return protect([&] {
        auto [mag, phase] = librosa::magphase(complex_matrix_from_js(d), real_prop(options, "power", 1));
        val out = val::object();
        out.set("magnitude", matrix_to_js(mag));
        out.set("phase", complex_matrix_to_js(phase));
        return out;
    });
}

val magnitude_js(val d) {
    return protect([&] {
        return matrix_to_js(librosa::magnitude(complex_matrix_from_js(d)));
    });
}

val phase_js(val d) {
    return protect([&] {
        return matrix_to_js(librosa::phase(complex_matrix_from_js(d)));
    });
}

val power_to_db_js(val s, val options) {
    return protect([&] {
        return matrix_to_js(librosa::power_to_db(matrix_from_js(s),
                                                 real_prop(options, "ref", 1),
                                                 real_prop(options, "amin", 1e-10),
                                                 opt_real_prop(options, "topDb", 80.0)));
    });
}

val power_to_db_scalar_js(val s, val options) {
    return protect([&] {
        return val(librosa::power_to_db(s.as<Real>(),
                                        real_prop(options, "ref", 1),
                                        real_prop(options, "amin", 1e-10),
                                        opt_real_prop(options, "topDb")));
    });
}

val db_to_power_js(val s, val options) {
    return protect([&] {
        if (is_number(s)) return val(librosa::db_to_power(s.as<Real>(), real_prop(options, "ref", 1)));
        return matrix_to_js(librosa::db_to_power(matrix_from_js(s), real_prop(options, "ref", 1)));
    });
}

val amplitude_to_db_js(val s, val options) {
    return protect([&] {
        if (is_number(s)) {
            return val(librosa::amplitude_to_db(s.as<Real>(), real_prop(options, "ref", 1),
                                                real_prop(options, "amin", 1e-5),
                                                opt_real_prop(options, "topDb")));
        }
        return matrix_to_js(librosa::amplitude_to_db(matrix_from_js(s),
                                                     real_prop(options, "ref", 1),
                                                     real_prop(options, "amin", 1e-5),
                                                     opt_real_prop(options, "topDb", 80.0)));
    });
}

val db_to_amplitude_js(val s, val options) {
    return protect([&] {
        if (is_number(s)) return val(librosa::db_to_amplitude(s.as<Real>(), real_prop(options, "ref", 1)));
        return matrix_to_js(librosa::db_to_amplitude(matrix_from_js(s), real_prop(options, "ref", 1)));
    });
}

val perceptual_weighting_js(val s, val frequencies, val options) {
    return protect([&] {
        return matrix_to_js(librosa::perceptual_weighting(matrix_from_js(s), vector_from_js(frequencies),
                                                          weight_type_from_string(string_prop(options, "kind", "A")),
                                                          real_prop(options, "ref", 1),
                                                          real_prop(options, "amin", 1e-10),
                                                          opt_real_prop(options, "topDb", 80.0)));
    });
}

val phase_vocoder_js(val d, val options) {
    return protect([&] {
        return complex_matrix_to_js(librosa::phase_vocoder(complex_matrix_from_js(d),
                                                           real_prop(options, "rate", 1),
                                                           opt_int_prop(options, "hopLength"),
                                                           opt_int_prop(options, "nFft")));
    });
}

val griffinlim_js(val s, val options) {
    return protect([&] {
        return vector_to_js(librosa::griffinlim(matrix_from_js(s),
                                                int_prop(options, "nIter", 32),
                                                opt_int_prop(options, "hopLength"),
                                                opt_int_prop(options, "winLength"),
                                                opt_int_prop(options, "nFft"),
                                                window_prop(options, "window", WindowType::Hann),
                                                bool_prop(options, "center", true),
                                                opt_int_prop(options, "length"),
                                                pad_mode_prop(options, "padMode", PadMode::Constant),
                                                real_prop(options, "momentum", 0.99),
                                                string_prop(options, "initPhase", "random"),
                                                opt_uint_from_optional_int(opt_int_prop(options, "randomState"))));
    });
}

val pcen_js(val s, val options) {
    return protect([&] {
        return matrix_to_js(librosa::pcen(matrix_from_js(s),
                                          real_prop(options, "sr", 22050),
                                          int_prop(options, "hopLength", 512),
                                          real_prop(options, "gain", 0.98),
                                          real_prop(options, "bias", 2),
                                          real_prop(options, "power", 0.5),
                                          real_prop(options, "timeConstant", 0.4),
                                          real_prop(options, "eps", 1e-6)));
    });
}

val get_window_js(val window, val options) {
    return protect([&] {
        int n = int_prop(options, "length", int_prop(options, "nFft", 2048));
        bool fftbins = bool_prop(options, "fftbins", true);
        return vector_to_js(librosa::get_window(window.as<std::string>(), n, fftbins));
    });
}

val window_sumsquare_js(val window, val options) {
    return protect([&] {
        return vector_to_js(librosa::window_sumsquare(vector_from_js(window),
                                                      int_prop(options, "nFrames", 1),
                                                      int_prop(options, "hopLength", 512),
                                                      int_prop(options, "nFft", 2048)));
    });
}

// ---------------------------------------------------------------------------
// Constant-Q and feature extraction
// ---------------------------------------------------------------------------

val cqt_js(val y, val options) {
    return protect([&] {
        return complex_matrix_to_js(librosa::cqt(vector_from_js(y),
                                                 real_prop(options, "sr", 22050),
                                                 int_prop(options, "hopLength", 512),
                                                 opt_real_prop(options, "fmin"),
                                                 int_prop(options, "nBins", 84),
                                                 int_prop(options, "binsPerOctave", 12),
                                                 opt_real_prop(options, "tuning", 0.0),
                                                 real_prop(options, "filterScale", 1),
                                                 opt_real_prop(options, "norm", 1.0),
                                                 real_prop(options, "sparsity", 0.01),
                                                 window_prop(options, "window", WindowType::Hann),
                                                 bool_prop(options, "scale", true),
                                                 pad_mode_prop(options, "padMode", PadMode::Constant)));
    });
}

val vqt_js(val y, val options) {
    return protect([&] {
        return complex_matrix_to_js(librosa::vqt(vector_from_js(y),
                                                 real_prop(options, "sr", 22050),
                                                 int_prop(options, "hopLength", 512),
                                                 opt_real_prop(options, "fmin"),
                                                 int_prop(options, "nBins", 84),
                                                 opt_real_prop(options, "gamma"),
                                                 int_prop(options, "binsPerOctave", 12),
                                                 opt_real_prop(options, "tuning", 0.0),
                                                 real_prop(options, "filterScale", 1),
                                                 opt_real_prop(options, "norm", 1.0),
                                                 real_prop(options, "sparsity", 0.01),
                                                 window_prop(options, "window", WindowType::Hann),
                                                 bool_prop(options, "scale", true),
                                                 pad_mode_prop(options, "padMode", PadMode::Constant)));
    });
}

val pseudo_cqt_js(val y, val options) {
    return protect([&] {
        return matrix_to_js(librosa::pseudo_cqt(vector_from_js(y),
                                                real_prop(options, "sr", 22050),
                                                int_prop(options, "hopLength", 512),
                                                opt_real_prop(options, "fmin"),
                                                int_prop(options, "nBins", 84),
                                                int_prop(options, "binsPerOctave", 12),
                                                opt_real_prop(options, "tuning", 0.0),
                                                real_prop(options, "filterScale", 1),
                                                opt_real_prop(options, "norm", 1.0),
                                                real_prop(options, "sparsity", 0.01),
                                                window_prop(options, "window", WindowType::Hann),
                                                bool_prop(options, "scale", true),
                                                pad_mode_prop(options, "padMode", PadMode::Constant)));
    });
}

val icqt_js(val c, val options) {
    return protect([&] {
        return vector_to_js(librosa::icqt(complex_matrix_from_js(c),
                                          real_prop(options, "sr", 22050),
                                          int_prop(options, "hopLength", 512),
                                          opt_real_prop(options, "fmin"),
                                          int_prop(options, "binsPerOctave", 12),
                                          real_prop(options, "tuning", 0),
                                          real_prop(options, "filterScale", 1),
                                          opt_real_prop(options, "norm", 1.0),
                                          real_prop(options, "sparsity", 0.01),
                                          window_prop(options, "window", WindowType::Hann),
                                          bool_prop(options, "scale", true),
                                          opt_int_prop(options, "length")));
    });
}

val griffinlim_cqt_js(val c, val options) {
    return protect([&] {
        return vector_to_js(librosa::griffinlim_cqt(matrix_from_js(c),
                                                    int_prop(options, "nIter", 32),
                                                    real_prop(options, "sr", 22050),
                                                    int_prop(options, "hopLength", 512),
                                                    opt_real_prop(options, "fmin"),
                                                    int_prop(options, "binsPerOctave", 12),
                                                    real_prop(options, "tuning", 0),
                                                    real_prop(options, "filterScale", 1),
                                                    opt_real_prop(options, "norm", 1.0),
                                                    real_prop(options, "sparsity", 0.01),
                                                    window_prop(options, "window", WindowType::Hann),
                                                    bool_prop(options, "scale", true),
                                                    pad_mode_prop(options, "padMode", PadMode::Constant),
                                                    opt_int_prop(options, "length"),
                                                    real_prop(options, "momentum", 0.99),
                                                    string_prop(options, "initPhase", "random"),
                                                    opt_uint_from_optional_int(opt_int_prop(options, "randomState"))));
    });
}

val melspectrogram_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::melspectrogram(matrix_from_js(input),
                                                                 real_prop(options, "sr", 22050),
                                                                 int_prop(options, "nFft", 2048),
                                                                 int_prop(options, "nMels", 128),
                                                                 real_prop(options, "fmin", 0),
                                                                 opt_real_prop(options, "fmax"),
                                                                 bool_prop(options, "htk", false),
                                                                 bool_prop(options, "normSlaney", true)));
        }
        return matrix_to_js(librosa::feature::melspectrogram(vector_from_js(input),
                                                             real_prop(options, "sr", 22050),
                                                             int_prop(options, "nFft", 2048),
                                                             int_prop(options, "hopLength", 512),
                                                             opt_int_prop(options, "winLength"),
                                                             window_prop(options, "window", WindowType::Hann),
                                                             bool_prop(options, "center", true),
                                                             pad_mode_prop(options, "padMode", PadMode::Constant),
                                                             real_prop(options, "power", 2),
                                                             int_prop(options, "nMels", 128),
                                                             real_prop(options, "fmin", 0),
                                                             opt_real_prop(options, "fmax"),
                                                             bool_prop(options, "htk", false),
                                                             bool_prop(options, "normSlaney", true)));
    });
}

val mfcc_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::mfcc(matrix_from_js(input),
                                                       int_prop(options, "nMfcc", 20),
                                                       int_prop(options, "dctType", 2),
                                                       bool_prop(options, "normOrtho", true),
                                                       real_prop(options, "lifter", 0)));
        }
        return matrix_to_js(librosa::feature::mfcc(vector_from_js(input),
                                                   real_prop(options, "sr", 22050),
                                                   int_prop(options, "nMfcc", 20),
                                                   int_prop(options, "dctType", 2),
                                                   bool_prop(options, "normOrtho", true),
                                                   real_prop(options, "lifter", 0),
                                                   int_prop(options, "nFft", 2048),
                                                   int_prop(options, "hopLength", 512),
                                                   int_prop(options, "nMels", 128),
                                                   real_prop(options, "fmin", 0),
                                                   opt_real_prop(options, "fmax"),
                                                   bool_prop(options, "htk", false)));
    });
}

val chroma_stft_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::chroma_stft(matrix_from_js(input),
                                                              real_prop(options, "sr", 22050),
                                                              int_prop(options, "nFft", 2048),
                                                              int_prop(options, "nChroma", 12),
                                                              opt_real_prop(options, "tuning"),
                                                              real_prop(options, "norm", std::numeric_limits<Real>::infinity())));
        }
        return matrix_to_js(librosa::feature::chroma_stft(vector_from_js(input),
                                                          real_prop(options, "sr", 22050),
                                                          int_prop(options, "nFft", 2048),
                                                          int_prop(options, "hopLength", 512),
                                                          int_prop(options, "nChroma", 12),
                                                          opt_real_prop(options, "tuning"),
                                                          real_prop(options, "norm", std::numeric_limits<Real>::infinity()),
                                                          window_prop(options, "window", WindowType::Hann),
                                                          bool_prop(options, "center", true)));
    });
}

val chroma_cqt_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::chroma_cqt(matrix_from_js(input),
                                                             opt_real_prop(options, "fmin"),
                                                             real_prop(options, "norm", std::numeric_limits<Real>::infinity()),
                                                             real_prop(options, "threshold", 0),
                                                             int_prop(options, "nChroma", 12),
                                                             int_prop(options, "binsPerOctave", 36)));
        }
        return matrix_to_js(librosa::feature::chroma_cqt(vector_from_js(input),
                                                         real_prop(options, "sr", 22050),
                                                         int_prop(options, "hopLength", 512),
                                                         opt_real_prop(options, "fmin"),
                                                         real_prop(options, "norm", std::numeric_limits<Real>::infinity()),
                                                         real_prop(options, "threshold", 0),
                                                         opt_real_prop(options, "tuning"),
                                                         int_prop(options, "nChroma", 12),
                                                         int_prop(options, "nOctaves", 7),
                                                         int_prop(options, "binsPerOctave", 36)));
    });
}

val chroma_cens_js(val y, val options) {
    return protect([&] {
        return matrix_to_js(librosa::feature::chroma_cens(vector_from_js(y),
                                                          real_prop(options, "sr", 22050),
                                                          int_prop(options, "hopLength", 512),
                                                          opt_real_prop(options, "fmin"),
                                                          opt_real_prop(options, "tuning"),
                                                          int_prop(options, "nChroma", 12),
                                                          int_prop(options, "nOctaves", 7),
                                                          int_prop(options, "binsPerOctave", 36),
                                                          real_prop(options, "norm", 2),
                                                          int_prop(options, "winLenSmooth", 41)));
    });
}

val chroma_vqt_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::chroma_vqt(matrix_from_js(input),
                                                             opt_real_prop(options, "fmin"),
                                                             real_prop(options, "norm", std::numeric_limits<Real>::infinity()),
                                                             real_prop(options, "threshold", 0),
                                                             int_prop(options, "binsPerOctave", 12)));
        }
        return matrix_to_js(librosa::feature::chroma_vqt(vector_from_js(input),
                                                         real_prop(options, "sr", 22050),
                                                         int_prop(options, "hopLength", 512),
                                                         opt_real_prop(options, "fmin"),
                                                         real_prop(options, "norm", std::numeric_limits<Real>::infinity()),
                                                         real_prop(options, "threshold", 0),
                                                         int_prop(options, "nOctaves", 7),
                                                         int_prop(options, "binsPerOctave", 12),
                                                         real_prop(options, "gamma", 0)));
    });
}

val spectral_centroid_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            val freq_v = prop(options, "freq");
            ArrayXr freq_storage;
            const ArrayXr* freq = nullptr;
            if (!is_nullish(freq_v)) {
                freq_storage = vector_from_js(freq_v);
                freq = &freq_storage;
            }
            return matrix_to_js(librosa::feature::spectral_centroid(matrix_from_js(input),
                                                                    real_prop(options, "sr", 22050),
                                                                    int_prop(options, "nFft", 2048),
                                                                    freq));
        }
        return matrix_to_js(librosa::feature::spectral_centroid(vector_from_js(input),
                                                                real_prop(options, "sr", 22050),
                                                                int_prop(options, "nFft", 2048),
                                                                int_prop(options, "hopLength", 512),
                                                                window_prop(options, "window", WindowType::Hann),
                                                                bool_prop(options, "center", true)));
    });
}

val spectral_bandwidth_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::spectral_bandwidth(matrix_from_js(input),
                                                                     real_prop(options, "sr", 22050),
                                                                     int_prop(options, "nFft", 2048),
                                                                     nullptr, nullptr,
                                                                     real_prop(options, "p", 2),
                                                                     bool_prop(options, "norm", true)));
        }
        return matrix_to_js(librosa::feature::spectral_bandwidth(vector_from_js(input),
                                                                 real_prop(options, "sr", 22050),
                                                                 int_prop(options, "nFft", 2048),
                                                                 int_prop(options, "hopLength", 512),
                                                                 window_prop(options, "window", WindowType::Hann),
                                                                 bool_prop(options, "center", true),
                                                                 real_prop(options, "p", 2),
                                                                 bool_prop(options, "norm", true)));
    });
}

val spectral_rolloff_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::spectral_rolloff(matrix_from_js(input),
                                                                   real_prop(options, "sr", 22050),
                                                                   int_prop(options, "nFft", 2048),
                                                                   nullptr,
                                                                   real_prop(options, "rollPercent", 0.85)));
        }
        return matrix_to_js(librosa::feature::spectral_rolloff(vector_from_js(input),
                                                               real_prop(options, "sr", 22050),
                                                               int_prop(options, "nFft", 2048),
                                                               int_prop(options, "hopLength", 512),
                                                               window_prop(options, "window", WindowType::Hann),
                                                               bool_prop(options, "center", true),
                                                               real_prop(options, "rollPercent", 0.85)));
    });
}

val spectral_flatness_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::spectral_flatness(matrix_from_js(input),
                                                                    real_prop(options, "amin", 1e-10),
                                                                    real_prop(options, "power", 2)));
        }
        return matrix_to_js(librosa::feature::spectral_flatness(vector_from_js(input),
                                                                int_prop(options, "nFft", 2048),
                                                                int_prop(options, "hopLength", 512),
                                                                window_prop(options, "window", WindowType::Hann),
                                                                bool_prop(options, "center", true),
                                                                real_prop(options, "amin", 1e-10),
                                                                real_prop(options, "power", 2)));
    });
}

val spectral_contrast_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::spectral_contrast(matrix_from_js(input),
                                                                    real_prop(options, "sr", 22050),
                                                                    int_prop(options, "nFft", 2048),
                                                                    nullptr,
                                                                    real_prop(options, "fmin", 200),
                                                                    int_prop(options, "nBands", 6),
                                                                    real_prop(options, "quantile", 0.02),
                                                                    bool_prop(options, "linear", false)));
        }
        return matrix_to_js(librosa::feature::spectral_contrast(vector_from_js(input),
                                                                real_prop(options, "sr", 22050),
                                                                int_prop(options, "nFft", 2048),
                                                                int_prop(options, "hopLength", 512),
                                                                window_prop(options, "window", WindowType::Hann),
                                                                bool_prop(options, "center", true),
                                                                real_prop(options, "fmin", 200),
                                                                int_prop(options, "nBands", 6),
                                                                real_prop(options, "quantile", 0.02),
                                                                bool_prop(options, "linear", false)));
    });
}

val rms_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::rms(matrix_from_js(input), int_prop(options, "frameLength", 2048)));
        }
        return matrix_to_js(librosa::feature::rms(vector_from_js(input),
                                                  int_prop(options, "frameLength", 2048),
                                                  int_prop(options, "hopLength", 512),
                                                  bool_prop(options, "center", true)));
    });
}

val zero_crossing_rate_js(val y, val options) {
    return protect([&] {
        return matrix_to_js(librosa::feature::zero_crossing_rate(vector_from_js(y),
                                                                 int_prop(options, "frameLength", 2048),
                                                                 int_prop(options, "hopLength", 512),
                                                                 bool_prop(options, "center", true),
                                                                 real_prop(options, "threshold", 0)));
    });
}

val poly_features_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::poly_features(matrix_from_js(input),
                                                                real_prop(options, "sr", 22050),
                                                                int_prop(options, "nFft", 2048),
                                                                int_prop(options, "order", 1),
                                                                nullptr));
        }
        return matrix_to_js(librosa::feature::poly_features(vector_from_js(input),
                                                            real_prop(options, "sr", 22050),
                                                            int_prop(options, "nFft", 2048),
                                                            int_prop(options, "hopLength", 512),
                                                            int_prop(options, "order", 1)));
    });
}

val tonnetz_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::feature::tonnetz(matrix_from_js(input)));
        }
        return matrix_to_js(librosa::feature::tonnetz(vector_from_js(input), real_prop(options, "sr", 22050), nullptr));
    });
}

val delta_js(val data, val options) {
    return protect([&] {
        return matrix_to_js(librosa::feature::delta(matrix_from_js(data),
                                                    int_prop(options, "width", 9),
                                                    int_prop(options, "order", 1),
                                                    int_prop(options, "axis", -1),
                                                    string_prop(options, "mode", "interp")));
    });
}

val stack_memory_js(val data, val options) {
    return protect([&] {
        return matrix_to_js(librosa::feature::stack_memory(matrix_from_js(data),
                                                           int_prop(options, "nSteps", 2),
                                                           int_prop(options, "delay", 1),
                                                           pad_mode_prop(options, "mode", PadMode::Constant)));
    });
}

val fourier_tempogram_js(val input, val options) {
    return protect([&] {
        if (bool_prop(options, "fromAudio", false)) {
            return complex_matrix_to_js(librosa::feature::fourier_tempogram_audio(vector_from_js(input),
                                                                                  real_prop(options, "sr", 22050),
                                                                                  int_prop(options, "hopLength", 512),
                                                                                  int_prop(options, "winLength", 384),
                                                                                  bool_prop(options, "center", true),
                                                                                  window_prop(options, "window", WindowType::Hann)));
        }
        return complex_matrix_to_js(librosa::feature::fourier_tempogram(vector_from_js(input),
                                                                        real_prop(options, "sr", 22050),
                                                                        int_prop(options, "hopLength", 512),
                                                                        int_prop(options, "winLength", 384),
                                                                        bool_prop(options, "center", true),
                                                                        window_prop(options, "window", WindowType::Hann)));
    });
}

val tempogram_ratio_js(val tg, val options) {
    return protect([&] {
        std::vector<Real> factors;
        val factors_v = prop(options, "factors");
        if (!is_nullish(factors_v)) factors = real_vector_from_js(factors_v);
        return matrix_to_js(librosa::feature::tempogram_ratio(matrix_from_js(tg),
                                                              real_prop(options, "sr", 22050),
                                                              int_prop(options, "hopLength", 512),
                                                              factors,
                                                              real_prop(options, "startBpm", 120),
                                                              real_prop(options, "stdBpm", 1),
                                                              opt_real_prop(options, "maxTempo", 320.0),
                                                              real_prop(options, "fillValue", 0)));
    });
}

val mel_to_stft_js(val m, val options) {
    return protect([&] {
        return matrix_to_js(librosa::feature::mel_to_stft(matrix_from_js(m),
                                                          real_prop(options, "sr", 22050),
                                                          int_prop(options, "nFft", 2048),
                                                          real_prop(options, "power", 2)));
    });
}

val mel_to_audio_js(val m, val options) {
    return protect([&] {
        return vector_to_js(librosa::feature::mel_to_audio(matrix_from_js(m),
                                                           real_prop(options, "sr", 22050),
                                                           int_prop(options, "nFft", 2048),
                                                           opt_int_prop(options, "hopLength"),
                                                           opt_int_prop(options, "winLength"),
                                                           window_prop(options, "window", WindowType::Hann),
                                                           bool_prop(options, "center", true),
                                                           real_prop(options, "power", 2),
                                                           int_prop(options, "nIter", 32),
                                                           opt_int_prop(options, "length")));
    });
}

val mfcc_to_mel_js(val m, val options) {
    return protect([&] {
        return matrix_to_js(librosa::feature::mfcc_to_mel(matrix_from_js(m),
                                                          int_prop(options, "nMels", 128),
                                                          int_prop(options, "dctType", 2),
                                                          bool_prop(options, "orthoNorm", true),
                                                          real_prop(options, "ref", 1),
                                                          real_prop(options, "lifter", 0)));
    });
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

val mel_filter_js(val options) {
    return protect([&] {
        return matrix_to_js(librosa::filters::mel(real_prop(options, "sr", 22050),
                                                  int_prop(options, "nFft", 2048),
                                                  int_prop(options, "nMels", 128),
                                                  real_prop(options, "fmin", 0),
                                                  opt_real_prop(options, "fmax"),
                                                  bool_prop(options, "htk", false),
                                                  mel_norm_from_string(string_prop(options, "norm", "slaney"))));
    });
}

val chroma_filter_js(val options) {
    return protect([&] {
        return matrix_to_js(librosa::filters::chroma(real_prop(options, "sr", 22050),
                                                     int_prop(options, "nFft", 2048),
                                                     int_prop(options, "nChroma", 12),
                                                     real_prop(options, "tuning", 0),
                                                     real_prop(options, "ctroct", 5),
                                                     opt_real_prop(options, "octwidth", 2.0),
                                                     opt_real_prop(options, "norm", 2.0),
                                                     bool_prop(options, "baseC", true)));
    });
}

val window_bandwidth_js(val window, val options) {
    return protect([&] {
        return val(librosa::filters::window_bandwidth(window.as<std::string>(), int_prop(options, "n", 1000)));
    });
}

val cq_to_chroma_js(val options) {
    return protect([&] {
        return matrix_to_js(librosa::filters::cq_to_chroma(int_prop(options, "nInput", 84),
                                                           int_prop(options, "binsPerOctave", 12),
                                                           int_prop(options, "nChroma", 12),
                                                           opt_real_prop(options, "fmin"),
                                                           bool_prop(options, "baseC", true)));
    });
}

val relative_bandwidth_js(val freqs) {
    return protect([&] {
        return vector_to_js(librosa::filters::relative_bandwidth(vector_from_js(freqs)));
    });
}

val wavelet_lengths_js(val freqs, val options) {
    return protect([&] {
        auto [lengths, cutoff] = librosa::filters::wavelet_lengths(vector_from_js(freqs),
                                                                   real_prop(options, "sr", 22050),
                                                                   window_prop(options, "window", WindowType::Hann),
                                                                   real_prop(options, "filterScale", 1),
                                                                   real_prop(options, "gamma", 0),
                                                                   opt_real_prop(options, "alpha"));
        val out = val::object();
        out.set("lengths", vector_to_js(lengths));
        out.set("cutoff", cutoff);
        return out;
    });
}

val wavelet_js(val freqs, val options) {
    return protect([&] {
        auto [filters, lengths] = librosa::filters::wavelet(vector_from_js(freqs),
                                                            real_prop(options, "sr", 22050),
                                                            window_prop(options, "window", WindowType::Hann),
                                                            real_prop(options, "filterScale", 1),
                                                            bool_prop(options, "padFft", true),
                                                            opt_real_prop(options, "norm", 1.0),
                                                            real_prop(options, "gamma", 0),
                                                            opt_real_prop(options, "alpha"));
        val out = val::object();
        out.set("filters", complex_matrix_to_js(filters));
        out.set("lengths", vector_to_js(lengths));
        return out;
    });
}

val diagonal_filter_js(val options) {
    return protect([&] {
        return matrix_to_js(librosa::filters::diagonal_filter(window_prop(options, "window", WindowType::Hann),
                                                              int_prop(options, "n", 7),
                                                              real_prop(options, "slope", 1),
                                                              opt_real_prop(options, "angle"),
                                                              bool_prop(options, "zeroMean", false)));
    });
}

val mr_frequencies_js(val options) {
    return protect([&] {
        auto [freqs, rates] = librosa::filters::mr_frequencies(real_prop(options, "tuning", 0));
        val out = val::object();
        out.set("frequencies", vector_to_js(freqs));
        out.set("sampleRates", vector_to_js(rates));
        return out;
    });
}

// ---------------------------------------------------------------------------
// Pitch
// ---------------------------------------------------------------------------

val pitch_tuning_js(val frequencies, val options) {
    return protect([&] {
        return val(librosa::pitch_tuning(vector_from_js(frequencies),
                                         real_prop(options, "resolution", 0.01),
                                         int_prop(options, "binsPerOctave", 12)));
    });
}

val estimate_tuning_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return val(librosa::estimate_tuning(matrix_from_js(input),
                                                real_prop(options, "sr", 22050),
                                                int_prop(options, "nFft", 2048),
                                                real_prop(options, "resolution", 0.01),
                                                int_prop(options, "binsPerOctave", 12),
                                                real_prop(options, "fmin", 150),
                                                real_prop(options, "fmax", 4000),
                                                real_prop(options, "threshold", 0.1)));
        }
        return val(librosa::estimate_tuning(vector_from_js(input),
                                            real_prop(options, "sr", 22050),
                                            int_prop(options, "nFft", 2048),
                                            real_prop(options, "resolution", 0.01),
                                            int_prop(options, "binsPerOctave", 12),
                                            real_prop(options, "fmin", 150),
                                            real_prop(options, "fmax", 4000),
                                            real_prop(options, "threshold", 0.1)));
    });
}

val piptrack_js(val input, val options) {
    return protect([&] {
        std::pair<ArrayXXr, ArrayXXr> result;
        if (!prop(input, "rows").isUndefined()) {
            result = librosa::piptrack(matrix_from_js(input),
                                       real_prop(options, "sr", 22050),
                                       int_prop(options, "nFft", 2048),
                                       opt_int_prop(options, "hopLength"),
                                       real_prop(options, "fmin", 150),
                                       real_prop(options, "fmax", 4000),
                                       real_prop(options, "threshold", 0.1));
        } else {
            result = librosa::piptrack(vector_from_js(input),
                                       real_prop(options, "sr", 22050),
                                       int_prop(options, "nFft", 2048),
                                       opt_int_prop(options, "hopLength"),
                                       real_prop(options, "fmin", 150),
                                       real_prop(options, "fmax", 4000),
                                       real_prop(options, "threshold", 0.1),
                                       bool_prop(options, "center", true));
        }
        val out = val::object();
        out.set("pitches", matrix_to_js(result.first));
        out.set("magnitudes", matrix_to_js(result.second));
        return out;
    });
}

val yin_js(val y, val options) {
    return protect([&] {
        return vector_to_js(librosa::yin(vector_from_js(y),
                                         real_prop(options, "fmin", 65),
                                         real_prop(options, "fmax", 2093),
                                         real_prop(options, "sr", 22050),
                                         int_prop(options, "frameLength", 2048),
                                         opt_int_prop(options, "hopLength"),
                                         real_prop(options, "troughThreshold", 0.1),
                                         bool_prop(options, "center", true),
                                         pad_mode_prop(options, "padMode", PadMode::Constant)));
    });
}

val pyin_js(val y, val options) {
    return protect([&] {
        auto result = librosa::pyin(vector_from_js(y),
                                    real_prop(options, "fmin", 65),
                                    real_prop(options, "fmax", 2093),
                                    real_prop(options, "sr", 22050),
                                    int_prop(options, "frameLength", 2048),
                                    opt_int_prop(options, "hopLength"),
                                    int_prop(options, "nThresholds", 100),
                                    real_prop(options, "betaA", 2),
                                    real_prop(options, "betaB", 18),
                                    real_prop(options, "boltzmannParameter", 2),
                                    real_prop(options, "resolution", 0.1),
                                    real_prop(options, "maxTransitionRate", 35.92),
                                    real_prop(options, "switchProb", 0.01),
                                    real_prop(options, "noTroughProb", 0.01),
                                    opt_real_prop(options, "fillNa", std::numeric_limits<Real>::quiet_NaN()),
                                    bool_prop(options, "center", true),
                                    pad_mode_prop(options, "padMode", PadMode::Constant));
        val out = val::object();
        out.set("f0", vector_to_js(result.f0));
        out.set("voicedFlag", bool_vector_to_js(result.voiced_flag));
        out.set("voicedProb", vector_to_js(result.voiced_prob));
        return out;
    });
}

// ---------------------------------------------------------------------------
// Beat and onset
// ---------------------------------------------------------------------------

val tempogram_js(val input, val options) {
    return protect([&] {
        if (bool_prop(options, "fromAudio", false)) {
            return matrix_to_js(librosa::beat::tempogram_audio(vector_from_js(input),
                                                               real_prop(options, "sr", 22050),
                                                               int_prop(options, "hopLength", 512),
                                                               int_prop(options, "winLength", 384),
                                                               bool_prop(options, "center", true),
                                                               window_prop(options, "window", WindowType::Hann),
                                                               opt_real_prop(options, "norm", std::numeric_limits<Real>::infinity())));
        }
        return matrix_to_js(librosa::beat::tempogram(vector_from_js(input),
                                                     real_prop(options, "sr", 22050),
                                                     int_prop(options, "hopLength", 512),
                                                     int_prop(options, "winLength", 384),
                                                     bool_prop(options, "center", true),
                                                     window_prop(options, "window", WindowType::Hann),
                                                     opt_real_prop(options, "norm", std::numeric_limits<Real>::infinity())));
    });
}

val tempo_js(val input, val options) {
    return protect([&] {
        if (bool_prop(options, "fromAudio", false)) {
            return val(librosa::beat::tempo_audio(vector_from_js(input),
                                                  real_prop(options, "sr", 22050),
                                                  int_prop(options, "hopLength", 512),
                                                  real_prop(options, "startBpm", 120),
                                                  real_prop(options, "stdBpm", 1),
                                                  real_prop(options, "acSize", 8),
                                                  opt_real_prop(options, "maxTempo", 320.0),
                                                  bool_prop(options, "aggregate", true)));
        }
        return val(librosa::beat::tempo(vector_from_js(input),
                                        real_prop(options, "sr", 22050),
                                        int_prop(options, "hopLength", 512),
                                        real_prop(options, "startBpm", 120),
                                        real_prop(options, "stdBpm", 1),
                                        real_prop(options, "acSize", 8),
                                        opt_real_prop(options, "maxTempo", 320.0),
                                        bool_prop(options, "aggregate", true)));
    });
}

val tempo_frames_js(val envelope, val options) {
    return protect([&] {
        return vector_to_js(librosa::beat::tempo_frames(vector_from_js(envelope),
                                                        real_prop(options, "sr", 22050),
                                                        int_prop(options, "hopLength", 512),
                                                        real_prop(options, "startBpm", 120),
                                                        real_prop(options, "stdBpm", 1),
                                                        real_prop(options, "acSize", 8),
                                                        opt_real_prop(options, "maxTempo", 320.0)));
    });
}

val beat_track_js(val input, val options) {
    return protect([&] {
        std::pair<Real, std::vector<Eigen::Index>> result;
        if (bool_prop(options, "fromAudio", false)) {
            result = librosa::beat::beat_track_audio(vector_from_js(input),
                                                     real_prop(options, "sr", 22050),
                                                     int_prop(options, "hopLength", 512),
                                                     real_prop(options, "startBpm", 120),
                                                     real_prop(options, "tightness", 100),
                                                     bool_prop(options, "trim", true),
                                                     opt_real_prop(options, "bpm"),
                                                     beat_units_from_string(string_prop(options, "units", "frames")));
        } else {
            result = librosa::beat::beat_track(vector_from_js(input),
                                               real_prop(options, "sr", 22050),
                                               int_prop(options, "hopLength", 512),
                                               real_prop(options, "startBpm", 120),
                                               real_prop(options, "tightness", 100),
                                               bool_prop(options, "trim", true),
                                               opt_real_prop(options, "bpm"),
                                               beat_units_from_string(string_prop(options, "units", "frames")));
        }
        val out = val::object();
        out.set("tempo", result.first);
        out.set("beats", index_vector_to_js(result.second));
        return out;
    });
}

val beat_track_times_js(val y, val options) {
    return protect([&] {
        auto [tempo, beats] = librosa::beat::beat_track_times(vector_from_js(y),
                                                              real_prop(options, "sr", 22050),
                                                              int_prop(options, "hopLength", 512),
                                                              real_prop(options, "startBpm", 120),
                                                              real_prop(options, "tightness", 100),
                                                              bool_prop(options, "trim", true));
        val out = val::object();
        out.set("tempo", tempo);
        out.set("beats", vector_to_js(beats));
        return out;
    });
}

val maximum_filter1d_js(val s, val options) {
    return protect([&] {
        return matrix_to_js(librosa::onset::maximum_filter1d(matrix_from_js(s),
                                                             int_prop(options, "size", 1),
                                                             int_prop(options, "axis", -2)));
    });
}

val match_events_js(val events_from, val events_to, val options) {
    return protect([&] {
        return index_vector_to_js(librosa::onset::match_events(index_vector_from_js(events_from),
                                                               index_vector_from_js(events_to),
                                                               bool_prop(options, "left", true),
                                                               bool_prop(options, "right", true)));
    });
}

val onset_strength_js(val input, val options) {
    return protect([&] {
        if (!prop(input, "rows").isUndefined()) {
            return vector_to_js(librosa::onset::onset_strength(matrix_from_js(input),
                                                               real_prop(options, "sr", 22050),
                                                               int_prop(options, "nFft", 2048),
                                                               int_prop(options, "hopLength", 512),
                                                               int_prop(options, "lag", 1),
                                                               int_prop(options, "maxSize", 1),
                                                               bool_prop(options, "detrend", false),
                                                               bool_prop(options, "center", true)));
        }
        return vector_to_js(librosa::onset::onset_strength(vector_from_js(input),
                                                           real_prop(options, "sr", 22050),
                                                           int_prop(options, "nFft", 2048),
                                                           int_prop(options, "hopLength", 512),
                                                           int_prop(options, "lag", 1),
                                                           int_prop(options, "maxSize", 1),
                                                           bool_prop(options, "detrend", false),
                                                           bool_prop(options, "center", true)));
    });
}

val onset_strength_multi_js(val input, val options) {
    return protect([&] {
        std::vector<int> channels;
        val ch = prop(options, "channels");
        if (!is_nullish(ch)) channels = int_vector_from_js(ch);
        if (!prop(input, "rows").isUndefined()) {
            return matrix_to_js(librosa::onset::onset_strength_multi(matrix_from_js(input),
                                                                     real_prop(options, "sr", 22050),
                                                                     int_prop(options, "nFft", 2048),
                                                                     int_prop(options, "hopLength", 512),
                                                                     int_prop(options, "lag", 1),
                                                                     int_prop(options, "maxSize", 1),
                                                                     bool_prop(options, "detrend", false),
                                                                     bool_prop(options, "center", true),
                                                                     channels));
        }
        return matrix_to_js(librosa::onset::onset_strength_multi(vector_from_js(input),
                                                                 real_prop(options, "sr", 22050),
                                                                 int_prop(options, "nFft", 2048),
                                                                 int_prop(options, "hopLength", 512),
                                                                 int_prop(options, "lag", 1),
                                                                 int_prop(options, "maxSize", 1),
                                                                 bool_prop(options, "detrend", false),
                                                                 bool_prop(options, "center", true),
                                                                 channels));
    });
}

val onset_detect_js(val input, val options) {
    return protect([&] {
        bool envelope = bool_prop(options, "envelope", false);
        std::vector<Eigen::Index> result = envelope
            ? librosa::onset::onset_detect_envelope(vector_from_js(input),
                                                    real_prop(options, "sr", 22050),
                                                    int_prop(options, "hopLength", 512),
                                                    bool_prop(options, "backtrack", false),
                                                    onset_units_from_string(string_prop(options, "units", "frames")),
                                                    bool_prop(options, "normalize", true),
                                                    int_prop(options, "preMax", 0),
                                                    int_prop(options, "postMax", 0),
                                                    int_prop(options, "preAvg", 0),
                                                    int_prop(options, "postAvg", 0),
                                                    real_prop(options, "delta", 0.07),
                                                    int_prop(options, "wait", 0))
            : librosa::onset::onset_detect(vector_from_js(input),
                                            real_prop(options, "sr", 22050),
                                            int_prop(options, "hopLength", 512),
                                            bool_prop(options, "backtrack", false),
                                            onset_units_from_string(string_prop(options, "units", "frames")),
                                            bool_prop(options, "normalize", true),
                                            int_prop(options, "preMax", 0),
                                            int_prop(options, "postMax", 0),
                                            int_prop(options, "preAvg", 0),
                                            int_prop(options, "postAvg", 0),
                                            real_prop(options, "delta", 0.07),
                                            int_prop(options, "wait", 0));
        return index_vector_to_js(result);
    });
}

val onset_detect_times_js(val y, val options) {
    return protect([&] {
        return vector_to_js(librosa::onset::onset_detect_times(vector_from_js(y),
                                                               real_prop(options, "sr", 22050),
                                                               int_prop(options, "hopLength", 512),
                                                               bool_prop(options, "backtrack", false),
                                                               bool_prop(options, "normalize", true)));
    });
}

val onset_backtrack_js(val events, val energy) {
    return protect([&] {
        return index_vector_to_js(librosa::onset::onset_backtrack(index_vector_from_js(events), vector_from_js(energy)));
    });
}

// ---------------------------------------------------------------------------
// Effects and decomposition
// ---------------------------------------------------------------------------

val time_stretch_js(val y, val options) {
    return protect([&] {
        return vector_to_js(librosa::effects::time_stretch(vector_from_js(y),
                                                           real_prop(options, "rate", 1),
                                                           int_prop(options, "nFft", 2048),
                                                           opt_int_prop(options, "hopLength"),
                                                           opt_int_prop(options, "winLength"),
                                                           window_prop(options, "window", WindowType::Hann),
                                                           bool_prop(options, "center", true)));
    });
}

val pitch_shift_js(val y, val options) {
    return protect([&] {
        return vector_to_js(librosa::effects::pitch_shift(vector_from_js(y),
                                                          real_prop(options, "sr", 22050),
                                                          real_prop(options, "nSteps", 0),
                                                          int_prop(options, "binsPerOctave", 12),
                                                          string_prop(options, "resType", "fft"),
                                                          int_prop(options, "nFft", 2048),
                                                          opt_int_prop(options, "hopLength")));
    });
}

val trim_js(val y, val options) {
    return protect([&] {
        auto [trimmed, index] = librosa::effects::trim(vector_from_js(y),
                                                       real_prop(options, "topDb", 60),
                                                       real_prop(options, "ref", -1),
                                                       int_prop(options, "frameLength", 2048),
                                                       int_prop(options, "hopLength", 512));
        val out = val::object();
        out.set("audio", vector_to_js(trimmed));
        std::vector<Eigen::Index> idx{index.first, index.second};
        out.set("index", index_vector_to_js(idx));
        return out;
    });
}

val split_js(val y, val options) {
    return protect([&] {
        return pair_index_vector_to_js(librosa::effects::split(vector_from_js(y),
                                                               real_prop(options, "topDb", 60),
                                                               real_prop(options, "ref", -1),
                                                               int_prop(options, "frameLength", 2048),
                                                               int_prop(options, "hopLength", 512)));
    });
}

val preemphasis_js(val y, val options) {
    return protect([&] {
        return vector_to_js(librosa::effects::preemphasis(vector_from_js(y), real_prop(options, "coef", 0.97)));
    });
}

val deemphasis_js(val y, val options) {
    return protect([&] {
        return vector_to_js(librosa::effects::deemphasis(vector_from_js(y), real_prop(options, "coef", 0.97)));
    });
}

val remix_js(val y, val intervals, val options) {
    return protect([&] {
        return vector_to_js(librosa::effects::remix(vector_from_js(y), intervals_from_js(intervals),
                                                    bool_prop(options, "alignZeros", true)));
    });
}

val harmonic_effect_js(val y, val options) {
    return protect([&] {
        return vector_to_js(librosa::effects::harmonic(vector_from_js(y),
                                                       int_prop(options, "kernelSize", 31),
                                                       real_prop(options, "power", 2),
                                                       bool_prop(options, "mask", false),
                                                       real_prop(options, "margin", 1),
                                                       int_prop(options, "nFft", 2048),
                                                       opt_int_prop(options, "hopLength"),
                                                       opt_int_prop(options, "winLength"),
                                                       window_prop(options, "window", WindowType::Hann),
                                                       bool_prop(options, "center", true),
                                                       pad_mode_prop(options, "padMode", PadMode::Constant)));
    });
}

val percussive_effect_js(val y, val options) {
    return protect([&] {
        return vector_to_js(librosa::effects::percussive(vector_from_js(y),
                                                         int_prop(options, "kernelSize", 31),
                                                         real_prop(options, "power", 2),
                                                         bool_prop(options, "mask", false),
                                                         real_prop(options, "margin", 1),
                                                         int_prop(options, "nFft", 2048),
                                                         opt_int_prop(options, "hopLength"),
                                                         opt_int_prop(options, "winLength"),
                                                         window_prop(options, "window", WindowType::Hann),
                                                         bool_prop(options, "center", true),
                                                         pad_mode_prop(options, "padMode", PadMode::Constant)));
    });
}

val median_filter_2d_js(val s, val options) {
    return protect([&] {
        val size_v = prop(options, "size");
        std::pair<int, int> size{3, 3};
        if (!is_nullish(size_v)) {
            if (is_number(size_v)) {
                size = {size_v.as<int>(), size_v.as<int>()};
            } else {
                size = {size_v[0].as<int>(), size_v[1].as<int>()};
            }
        }
        return matrix_to_js(librosa::decompose::median_filter_2d(matrix_from_js(s), size,
                                                                 string_prop(options, "mode", "reflect")));
    });
}

val hpss_js(val s, val options) {
    return protect([&] {
        bool complex = bool_prop(options, "complex", false);
        val kernel_v = prop(options, "kernelSize");
        val margin_v = prop(options, "margin");
        val out = val::object();
        if (complex) {
            std::pair<ArrayXXc, ArrayXXc> result;
            if (!is_nullish(kernel_v) && !is_number(kernel_v)) {
                std::pair<int, int> kernel{kernel_v[0].as<int>(), kernel_v[1].as<int>()};
                if (!is_nullish(margin_v) && !is_number(margin_v)) {
                    result = librosa::decompose::hpss_complex(complex_matrix_from_js(s), kernel,
                                                              real_prop(options, "power", 2),
                                                              bool_prop(options, "mask", false),
                                                              {margin_v[0].as<Real>(), margin_v[1].as<Real>()});
                } else {
                    result = librosa::decompose::hpss_complex(complex_matrix_from_js(s), kernel,
                                                              real_prop(options, "power", 2),
                                                              bool_prop(options, "mask", false),
                                                              real_prop(options, "margin", 1));
                }
            } else {
                result = librosa::decompose::hpss_complex(complex_matrix_from_js(s),
                                                          int_prop(options, "kernelSize", 31),
                                                          real_prop(options, "power", 2),
                                                          bool_prop(options, "mask", false),
                                                          real_prop(options, "margin", 1));
            }
            out.set("harmonic", complex_matrix_to_js(result.first));
            out.set("percussive", complex_matrix_to_js(result.second));
        } else {
            std::pair<ArrayXXr, ArrayXXr> result;
            if (!is_nullish(kernel_v) && !is_number(kernel_v)) {
                std::pair<int, int> kernel{kernel_v[0].as<int>(), kernel_v[1].as<int>()};
                if (!is_nullish(margin_v) && !is_number(margin_v)) {
                    result = librosa::decompose::hpss(matrix_from_js(s), kernel,
                                                      real_prop(options, "power", 2),
                                                      bool_prop(options, "mask", false),
                                                      {margin_v[0].as<Real>(), margin_v[1].as<Real>()});
                } else {
                    result = librosa::decompose::hpss(matrix_from_js(s), kernel,
                                                      real_prop(options, "power", 2),
                                                      bool_prop(options, "mask", false),
                                                      real_prop(options, "margin", 1));
                }
            } else {
                result = librosa::decompose::hpss(matrix_from_js(s),
                                                  int_prop(options, "kernelSize", 31),
                                                  real_prop(options, "power", 2),
                                                  bool_prop(options, "mask", false),
                                                  real_prop(options, "margin", 1));
            }
            out.set("harmonic", matrix_to_js(result.first));
            out.set("percussive", matrix_to_js(result.second));
        }
        return out;
    });
}

val decompose_nmf_js(val s, val options) {
    return protect([&] {
        auto [components, activations] = librosa::decompose::decompose_nmf(matrix_from_js(s),
                                                                           int_prop(options, "nComponents", 0),
                                                                           bool_prop(options, "sort", false),
                                                                           int_prop(options, "maxIter", 200),
                                                                           real_prop(options, "tol", 1e-4));
        val out = val::object();
        out.set("components", matrix_to_js(components));
        out.set("activations", matrix_to_js(activations));
        return out;
    });
}

// ---------------------------------------------------------------------------
// Sequence and segmentation
// ---------------------------------------------------------------------------

val transition_uniform_js(val options) {
    return protect([&] {
        return matrix_to_js(librosa::sequence::transition_uniform(int_prop(options, "nStates", 1)));
    });
}

val transition_loop_js(val prob, val options) {
    return protect([&] {
        int n_states = int_prop(options, "nStates", 1);
        if (is_number(prob)) return matrix_to_js(librosa::sequence::transition_loop(n_states, prob.as<Real>()));
        return matrix_to_js(librosa::sequence::transition_loop(n_states, vector_from_js(prob)));
    });
}

val transition_cycle_js(val prob, val options) {
    return protect([&] {
        int n_states = int_prop(options, "nStates", 1);
        if (is_number(prob)) return matrix_to_js(librosa::sequence::transition_cycle(n_states, prob.as<Real>()));
        return matrix_to_js(librosa::sequence::transition_cycle(n_states, vector_from_js(prob)));
    });
}

val transition_local_js(val options) {
    return protect([&] {
        return matrix_to_js(librosa::sequence::transition_local(int_prop(options, "nStates", 1),
                                                                int_prop(options, "width", 3),
                                                                window_prop(options, "window", WindowType::Triangle),
                                                                bool_prop(options, "wrap", false)));
    });
}

val viterbi_js(val prob, val transition, val options) {
    return protect([&] {
        val init_v = prop(options, "pInit");
        std::optional<ArrayXr> init = std::nullopt;
        if (!is_nullish(init_v)) init = vector_from_js(init_v);
        bool return_logp = bool_prop(options, "returnLogp", false);
        if (return_logp) {
            auto [states, logp] = librosa::sequence::viterbi_with_logp(matrix_from_js(prob), matrix_from_js(transition), init);
            val out = val::object();
            out.set("states", int_vector_to_js(states));
            out.set("logp", logp);
            return out;
        }
        return int_vector_to_js(librosa::sequence::viterbi(matrix_from_js(prob), matrix_from_js(transition), init));
    });
}

val viterbi_discriminative_js(val prob, val transition, val options) {
    return protect([&] {
        val state_v = prop(options, "pState");
        val init_v = prop(options, "pInit");
        std::optional<ArrayXr> state = std::nullopt;
        std::optional<ArrayXr> init = std::nullopt;
        if (!is_nullish(state_v)) state = vector_from_js(state_v);
        if (!is_nullish(init_v)) init = vector_from_js(init_v);
        return int_vector_to_js(librosa::sequence::viterbi_discriminative(matrix_from_js(prob),
                                                                          matrix_from_js(transition),
                                                                          state, init));
    });
}

val viterbi_binary_js(val prob, val transition, val options) {
    return protect([&] {
        val state_v = prop(options, "pState");
        val init_v = prop(options, "pInit");
        std::optional<ArrayXr> state = std::nullopt;
        std::optional<ArrayXr> init = std::nullopt;
        if (!is_nullish(state_v)) state = vector_from_js(state_v);
        if (!is_nullish(init_v)) init = vector_from_js(init_v);
        if (bool_prop(options, "returnLogp", false)) {
            auto [states, logp] = librosa::sequence::viterbi_binary_with_logp(matrix_from_js(prob),
                                                                              matrix_from_js(transition),
                                                                              state, init);
            val out = val::object();
            out.set("states", matrix_to_js(states));
            out.set("logp", vector_to_js(logp));
            return out;
        }
        return matrix_to_js(librosa::sequence::viterbi_binary(matrix_from_js(prob), matrix_from_js(transition), state, init));
    });
}

val cdist_euclidean_js(val x, val y) {
    return protect([&] {
        return matrix_to_js(librosa::sequence::cdist_euclidean(matrix_from_js(x), matrix_from_js(y)));
    });
}

val cdist_cosine_js(val x, val y) {
    return protect([&] {
        return matrix_to_js(librosa::sequence::cdist_cosine(matrix_from_js(x), matrix_from_js(y)));
    });
}

val dtw_js(val x, val y, val options) {
    return protect([&] {
        if (bool_prop(options, "backtrack", false)) {
            auto [cost, path] = librosa::sequence::dtw_backtrack(matrix_from_js(x), matrix_from_js(y),
                                                                 bool_prop(options, "subseq", false));
            val out = val::object();
            out.set("cost", matrix_to_js(cost));
            out.set("path", path_to_js(path));
            return out;
        }
        return matrix_to_js(librosa::sequence::dtw(matrix_from_js(x), matrix_from_js(y),
                                                   bool_prop(options, "subseq", false)));
    });
}

val rqa_js(val sim, val options) {
    return protect([&] {
        if (bool_prop(options, "backtrack", false)) {
            auto [score, path] = librosa::sequence::rqa_backtrack(matrix_from_js(sim),
                                                                  real_prop(options, "gapOnset", 1),
                                                                  real_prop(options, "gapExtend", 1),
                                                                  bool_prop(options, "knightMoves", true));
            val out = val::object();
            out.set("score", matrix_to_js(score));
            out.set("path", path_to_js(path));
            return out;
        }
        return matrix_to_js(librosa::sequence::rqa(matrix_from_js(sim),
                                                   real_prop(options, "gapOnset", 1),
                                                   real_prop(options, "gapExtend", 1),
                                                   bool_prop(options, "knightMoves", true)));
    });
}

val recurrence_to_lag_js(val rec, val options) {
    return protect([&] {
        return matrix_to_js(librosa::segment::recurrence_to_lag(matrix_from_js(rec),
                                                                bool_prop(options, "pad", true),
                                                                int_prop(options, "axis", -1)));
    });
}

val lag_to_recurrence_js(val lag, val options) {
    return protect([&] {
        return matrix_to_js(librosa::segment::lag_to_recurrence(matrix_from_js(lag),
                                                                int_prop(options, "axis", -1)));
    });
}

val cross_similarity_js(val data, val data_ref, val options) {
    return protect([&] {
        return matrix_to_js(librosa::segment::cross_similarity(matrix_from_js(data), matrix_from_js(data_ref),
                                                               int_prop(options, "k", 0),
                                                               string_prop(options, "metric", "euclidean"),
                                                               string_prop(options, "mode", "connectivity"),
                                                               real_prop(options, "bandwidth", 0)));
    });
}

val recurrence_matrix_js(val data, val options) {
    return protect([&] {
        return matrix_to_js(librosa::segment::recurrence_matrix(matrix_from_js(data),
                                                                int_prop(options, "k", 0),
                                                                int_prop(options, "width", 1),
                                                                string_prop(options, "metric", "euclidean"),
                                                                bool_prop(options, "sym", false),
                                                                string_prop(options, "mode", "connectivity"),
                                                                real_prop(options, "bandwidth", 0),
                                                                bool_prop(options, "self", false)));
    });
}

val path_enhance_js(val r, val options) {
    return protect([&] {
        return matrix_to_js(librosa::segment::path_enhance(matrix_from_js(r),
                                                           int_prop(options, "n", 7),
                                                           window_prop(options, "window", WindowType::Hann),
                                                           real_prop(options, "maxRatio", 2),
                                                           real_prop(options, "minRatio", 0),
                                                           int_prop(options, "nFilters", 7),
                                                           bool_prop(options, "zeroMean", false),
                                                           bool_prop(options, "clip", true)));
    });
}

val agglomerative_js(val data, val options) {
    return protect([&] {
        return index_vector_to_js(librosa::segment::agglomerative(matrix_from_js(data),
                                                                  int_prop(options, "k", 1),
                                                                  int_prop(options, "axis", -1)));
    });
}

val subsegment_js(val data, val frames, val options) {
    return protect([&] {
        return index_vector_to_js(librosa::segment::subsegment(matrix_from_js(data),
                                                               index_vector_from_js(frames),
                                                               int_prop(options, "nSegments", 4),
                                                               int_prop(options, "axis", -1)));
    });
}

// ---------------------------------------------------------------------------
// Harmonic helpers, util, notation, intervals
// ---------------------------------------------------------------------------

val interp_harmonics_js(val x, val freqs, val harmonics, val options) {
    return protect([&] {
        if (!prop(x, "rows").isUndefined()) {
            return matrix_to_js(librosa::core::interp_harmonics(matrix_from_js(x), vector_from_js(freqs),
                                                                real_vector_from_js(harmonics),
                                                                real_prop(options, "fillValue", 0)));
        }
        return matrix_to_js(librosa::core::interp_harmonics(vector_from_js(x), vector_from_js(freqs),
                                                            real_vector_from_js(harmonics),
                                                            real_prop(options, "fillValue", 0)));
    });
}

val f0_harmonics_js(val x, val f0, val freqs, val harmonics, val options) {
    return protect([&] {
        return matrix_to_js(librosa::core::f0_harmonics(matrix_from_js(x), vector_from_js(f0),
                                                        vector_from_js(freqs), real_vector_from_js(harmonics),
                                                        real_prop(options, "fillValue", 0)));
    });
}

val salience_js(val s, val freqs, val harmonics, val options) {
    return protect([&] {
        std::vector<Real> weights;
        val weights_v = prop(options, "weights");
        if (!is_nullish(weights_v)) weights = real_vector_from_js(weights_v);
        return matrix_to_js(librosa::core::salience(matrix_from_js(s), vector_from_js(freqs),
                                                    real_vector_from_js(harmonics), weights,
                                                    bool_prop(options, "filterPeaks", true),
                                                    real_prop(options, "fillValue", 0)));
    });
}

val frame_js(val x, val options) {
    return protect([&] {
        return matrix_to_js(librosa::util::frame(vector_from_js(x),
                                                 index_prop(options, "frameLength", 2048),
                                                 index_prop(options, "hopLength", 512)));
    });
}

val pad_center_js(val data, val options) {
    return protect([&] {
        if (!prop(data, "rows").isUndefined()) {
            return matrix_to_js(librosa::util::pad_center(matrix_from_js(data),
                                                          index_prop(options, "size", 0),
                                                          int_prop(options, "axis", -1),
                                                          pad_mode_prop(options, "mode", PadMode::Constant),
                                                          real_prop(options, "constantValue", 0)));
        }
        return vector_to_js(librosa::util::pad_center(vector_from_js(data),
                                                      index_prop(options, "size", 0),
                                                      pad_mode_prop(options, "mode", PadMode::Constant),
                                                      real_prop(options, "constantValue", 0)));
    });
}

val fix_length_js(val data, val options) {
    return protect([&] {
        if (!prop(data, "rows").isUndefined()) {
            return matrix_to_js(librosa::util::fix_length(matrix_from_js(data),
                                                          index_prop(options, "size", 0),
                                                          int_prop(options, "axis", -1),
                                                          pad_mode_prop(options, "mode", PadMode::Constant)));
        }
        return vector_to_js(librosa::util::fix_length(vector_from_js(data),
                                                      index_prop(options, "size", 0),
                                                      pad_mode_prop(options, "mode", PadMode::Constant)));
    });
}

val fix_frames_js(val frames, val options) {
    return protect([&] {
        auto xmax = opt_index_prop(options, "xMax");
        std::optional<Eigen::Index> xmax_eigen = std::nullopt;
        if (xmax) xmax_eigen = static_cast<Eigen::Index>(*xmax);
        return index_vector_to_js(librosa::util::fix_frames(index_vector_from_js(frames),
                                                            static_cast<Eigen::Index>(index_prop(options, "xMin", 0)),
                                                            xmax_eigen,
                                                            bool_prop(options, "pad", true)));
    });
}

val normalize_js(val data, val options) {
    return protect([&] {
        if (!prop(data, "rows").isUndefined()) {
            return matrix_to_js(librosa::util::normalize(matrix_from_js(data),
                                                         real_prop(options, "norm", std::numeric_limits<Real>::infinity()),
                                                         int_prop(options, "axis", 0),
                                                         opt_real_prop(options, "threshold"),
                                                         opt_bool_prop(options, "fill")));
        }
        return vector_to_js(librosa::util::normalize(vector_from_js(data),
                                                     real_prop(options, "norm", std::numeric_limits<Real>::infinity()),
                                                     opt_real_prop(options, "threshold"),
                                                     opt_bool_prop(options, "fill")));
    });
}

val localmax_js(val data, val options) {
    return protect([&] {
        if (!prop(data, "rows").isUndefined()) {
            return bool_matrix_to_js(librosa::util::localmax(matrix_from_js(data), int_prop(options, "axis", 0)));
        }
        return bool_vector_to_js(librosa::util::localmax(vector_from_js(data)));
    });
}

val localmin_js(val data, val options) {
    return protect([&] {
        if (!prop(data, "rows").isUndefined()) {
            return bool_matrix_to_js(librosa::util::localmin(matrix_from_js(data), int_prop(options, "axis", 0)));
        }
        return bool_vector_to_js(librosa::util::localmin(vector_from_js(data)));
    });
}

val peak_pick_js(val x, val options) {
    return protect([&] {
        return index_vector_to_js(librosa::util::peak_pick(vector_from_js(x),
                                                           int_prop(options, "preMax", 1),
                                                           int_prop(options, "postMax", 1),
                                                           int_prop(options, "preAvg", 1),
                                                           int_prop(options, "postAvg", 1),
                                                           real_prop(options, "delta", 0),
                                                           int_prop(options, "wait", 0)));
    });
}

val softmask_js(val x, val x_ref, val options) {
    return protect([&] {
        return matrix_to_js(librosa::util::softmask(matrix_from_js(x), matrix_from_js(x_ref),
                                                    real_prop(options, "power", 1),
                                                    bool_prop(options, "splitZeros", false)));
    });
}

val abs2_js(val x) {
    return protect([&] {
        if (!prop(x, "rows").isUndefined()) {
            return matrix_to_js(librosa::util::abs2(complex_matrix_from_js(x)));
        }
        return vector_to_js(librosa::util::abs2(complex_vector_from_js(x)));
    });
}

val phasor_js(val angles, val options) {
    return protect([&] {
        val mag = prop(options, "mag");
        if (!is_nullish(mag) && !is_number(mag)) {
            return complex_vector_to_js(librosa::util::phasor(vector_from_js(angles), vector_from_js(mag)));
        }
        return complex_vector_to_js(librosa::util::phasor(vector_from_js(angles), real_prop(options, "mag", 1)));
    });
}

val fill_off_diagonal_js(val x, val options) {
    return protect([&] {
        return matrix_to_js(librosa::util::fill_off_diagonal(matrix_from_js(x),
                                                             real_prop(options, "fillValue", 0),
                                                             bool_prop(options, "wrap", false)));
    });
}

val stack_js(val arrays, val options) {
    return protect([&] {
        const unsigned length = arrays["length"].as<unsigned>();
        std::vector<ArrayXr> values;
        values.reserve(length);
        for (unsigned i = 0; i < length; ++i) values.push_back(vector_from_js(arrays[i]));
        return matrix_to_js(librosa::util::stack(values, int_prop(options, "axis", 0)));
    });
}

val valid_audio_js(val y) {
    return protect([&] {
        if (!prop(y, "rows").isUndefined()) return val(librosa::util::valid_audio(matrix_from_js(y)));
        return val(librosa::util::valid_audio(vector_from_js(y)));
    });
}

val is_positive_int_js(val x) {
    return protect([&] { return val(librosa::util::is_positive_int(x.as<int>())); });
}

val valid_int_js(val x, val options) {
    return protect([&] { return val(librosa::util::valid_int(x.as<Real>(), bool_prop(options, "useFloor", true))); });
}

val pythagorean_intervals_js(val options) {
    return protect([&] {
        return vector_to_js(librosa::pythagorean_intervals(int_prop(options, "binsPerOctave", 12),
                                                           bool_prop(options, "sort", true)));
    });
}

val plimit_intervals_js(val primes, val options) {
    return protect([&] {
        return vector_to_js(librosa::plimit_intervals(int_vector_from_js(primes),
                                                      int_prop(options, "binsPerOctave", 12),
                                                      bool_prop(options, "sort", true)));
    });
}

val interval_frequencies_js(val intervals, val options) {
    return protect([&] {
        if (type_of(intervals) == "string") {
            return vector_to_js(librosa::interval_frequencies(int_prop(options, "nBins", 12),
                                                              real_prop(options, "fmin", 440),
                                                              intervals.as<std::string>(),
                                                              int_prop(options, "binsPerOctave", 12),
                                                              real_prop(options, "tuning", 0),
                                                              bool_prop(options, "sort", true)));
        }
        return vector_to_js(librosa::interval_frequencies(int_prop(options, "nBins", 12),
                                                          real_prop(options, "fmin", 440),
                                                          vector_from_js(intervals),
                                                          bool_prop(options, "sort", true)));
    });
}

val notation_int_vector_js(const std::vector<int>& values) {
    return int_std_vector_to_js(values);
}

val thaat_to_degrees_js(val thaat) {
    return protect([&] { return notation_int_vector_js(librosa::thaat_to_degrees(thaat.as<std::string>())); });
}

val mela_to_degrees_js(val mela) {
    return protect([&] {
        if (is_number(mela)) return notation_int_vector_js(librosa::mela_to_degrees(mela.as<int>()));
        return notation_int_vector_js(librosa::mela_to_degrees(mela.as<std::string>()));
    });
}

val mela_to_svara_js(val mela, val options) {
    return protect([&] {
        bool abbr = bool_prop(options, "abbr", true);
        bool unicode = bool_prop(options, "unicode", true);
        if (is_number(mela)) return string_vector_to_js(librosa::mela_to_svara(mela.as<int>(), abbr, unicode));
        return string_vector_to_js(librosa::mela_to_svara(mela.as<std::string>(), abbr, unicode));
    });
}

val list_mela_js() {
    return protect([&] { return map_string_int_to_js(librosa::list_mela()); });
}

val list_thaat_js() {
    return protect([&] { return string_vector_to_js(librosa::list_thaat()); });
}

val key_to_degrees_js(val key) {
    return protect([&] { return notation_int_vector_js(librosa::key_to_degrees(key.as<std::string>())); });
}

val key_to_notes_js(val key, val options) {
    return protect([&] { return string_vector_to_js(librosa::key_to_notes(key.as<std::string>(), bool_prop(options, "unicode", true))); });
}

val fifths_to_note_js(val unison, val fifths, val options) {
    return protect([&] {
        return val(librosa::fifths_to_note(unison.as<std::string>(), fifths.as<int>(),
                                           bool_prop(options, "unicode", true)));
    });
}

val interval_to_fjs_js(val interval, val options) {
    return protect([&] {
        return val(librosa::interval_to_fjs(interval.as<Real>(),
                                            string_prop(options, "unison", "C"),
                                            real_prop(options, "tolerance", 65.0 / 63.0),
                                            bool_prop(options, "unicode", true)));
    });
}

val midi_to_svara_h_js(val midi, val sa, val options) {
    return protect([&] {
        return val(librosa::midi_to_svara_h(midi.as<Real>(), sa.as<Real>(),
                                            bool_prop(options, "abbr", true),
                                            bool_prop(options, "octave", true),
                                            bool_prop(options, "unicode", true)));
    });
}

val midi_to_svara_c_js(val midi, val sa, val mela, val options) {
    return protect([&] {
        if (is_number(mela)) {
            return val(librosa::midi_to_svara_c(midi.as<Real>(), sa.as<Real>(), mela.as<int>(),
                                                bool_prop(options, "abbr", true),
                                                bool_prop(options, "octave", true),
                                                bool_prop(options, "unicode", true)));
        }
        return val(librosa::midi_to_svara_c(midi.as<Real>(), sa.as<Real>(), mela.as<std::string>(),
                                            bool_prop(options, "abbr", true),
                                            bool_prop(options, "octave", true),
                                            bool_prop(options, "unicode", true)));
    });
}

val hz_to_fjs_js(val frequency, val options) {
    return protect([&] {
        return val(librosa::hz_to_fjs(frequency.as<Real>(), real_prop(options, "fmin", 0),
                                      string_prop(options, "unison", ""),
                                      bool_prop(options, "unicode", true)));
    });
}

} // namespace

EMSCRIPTEN_BINDINGS(librosa_wasm) {
    function("buildInfo", &build_info);

    function("framesToSamples", &frames_to_samples_js);
    function("samplesToFrames", &samples_to_frames_js);
    function("framesToTime", &frames_to_time_js);
    function("timeToFrames", &time_to_frames_js);
    function("timeToSamples", &time_to_samples_js);
    function("samplesToTime", &samples_to_time_js);
    function("midiToHz", &midi_to_hz_js);
    function("hzToMidi", &hz_to_midi_js);
    function("noteToMidi", &note_to_midi_js);
    function("noteToHz", &note_to_hz_js);
    function("midiToNote", &midi_to_note_js);
    function("hzToNote", &hz_to_note_js);
    function("hzToMel", &hz_to_mel_js);
    function("melToHz", &mel_to_hz_js);
    function("hzToOcts", &hz_to_octs_js);
    function("octsToHz", &octs_to_hz_js);
    function("a4ToTuning", &a4_to_tuning_js);
    function("tuningToA4", &tuning_to_a4_js);
    function("fftFrequencies", &fft_frequencies_js);
    function("cqtFrequencies", &cqt_frequencies_js);
    function("melFrequencies", &mel_frequencies_js);
    function("tempoFrequencies", &tempo_frequencies_js);
    function("fourierTempoFrequencies", &fourier_tempo_frequencies_js);
    function("weighting", &weighting_js);

    function("toMono", &to_mono_js);
    function("resample", &resample_js);
    function("autocorrelate", &autocorrelate_js);
    function("lpc", &lpc_js);
    function("zeroCrossings", &zero_crossings_js);
    function("clicks", &clicks_js);
    function("clicksFrames", &clicks_frames_js);
    function("tone", &tone_js);
    function("chirp", &chirp_js);
    function("muCompress", &mu_compress_js);
    function("muExpand", &mu_expand_js);
    function("duration", &duration_js);

    function("stft", &stft_js);
    function("istft", &istft_js);
    function("magphase", &magphase_js);
    function("magnitude", &magnitude_js);
    function("phase", &phase_js);
    function("powerToDb", &power_to_db_js);
    function("powerToDbScalar", &power_to_db_scalar_js);
    function("dbToPower", &db_to_power_js);
    function("amplitudeToDb", &amplitude_to_db_js);
    function("dbToAmplitude", &db_to_amplitude_js);
    function("perceptualWeighting", &perceptual_weighting_js);
    function("phaseVocoder", &phase_vocoder_js);
    function("griffinlim", &griffinlim_js);
    function("pcen", &pcen_js);
    function("getWindow", &get_window_js);
    function("windowSumsquare", &window_sumsquare_js);

    function("cqt", &cqt_js);
    function("vqt", &vqt_js);
    function("pseudoCqt", &pseudo_cqt_js);
    function("icqt", &icqt_js);
    function("griffinlimCqt", &griffinlim_cqt_js);

    function("melspectrogram", &melspectrogram_js);
    function("mfcc", &mfcc_js);
    function("chromaStft", &chroma_stft_js);
    function("chromaCqt", &chroma_cqt_js);
    function("chromaCens", &chroma_cens_js);
    function("chromaVqt", &chroma_vqt_js);
    function("spectralCentroid", &spectral_centroid_js);
    function("spectralBandwidth", &spectral_bandwidth_js);
    function("spectralRolloff", &spectral_rolloff_js);
    function("spectralFlatness", &spectral_flatness_js);
    function("spectralContrast", &spectral_contrast_js);
    function("rms", &rms_js);
    function("zeroCrossingRate", &zero_crossing_rate_js);
    function("polyFeatures", &poly_features_js);
    function("tonnetz", &tonnetz_js);
    function("delta", &delta_js);
    function("stackMemory", &stack_memory_js);
    function("fourierTempogram", &fourier_tempogram_js);
    function("tempogramRatio", &tempogram_ratio_js);
    function("melToStft", &mel_to_stft_js);
    function("melToAudio", &mel_to_audio_js);
    function("mfccToMel", &mfcc_to_mel_js);

    function("melFilter", &mel_filter_js);
    function("chromaFilter", &chroma_filter_js);
    function("windowBandwidth", &window_bandwidth_js);
    function("cqToChroma", &cq_to_chroma_js);
    function("relativeBandwidth", &relative_bandwidth_js);
    function("waveletLengths", &wavelet_lengths_js);
    function("wavelet", &wavelet_js);
    function("diagonalFilter", &diagonal_filter_js);
    function("mrFrequencies", &mr_frequencies_js);

    function("pitchTuning", &pitch_tuning_js);
    function("estimateTuning", &estimate_tuning_js);
    function("piptrack", &piptrack_js);
    function("yin", &yin_js);
    function("pyin", &pyin_js);

    function("tempogram", &tempogram_js);
    function("tempo", &tempo_js);
    function("tempoFrames", &tempo_frames_js);
    function("beatTrack", &beat_track_js);
    function("beatTrackTimes", &beat_track_times_js);
    function("maximumFilter1d", &maximum_filter1d_js);
    function("matchEvents", &match_events_js);
    function("onsetStrength", &onset_strength_js);
    function("onsetStrengthMulti", &onset_strength_multi_js);
    function("onsetDetect", &onset_detect_js);
    function("onsetDetectTimes", &onset_detect_times_js);
    function("onsetBacktrack", &onset_backtrack_js);

    function("timeStretch", &time_stretch_js);
    function("pitchShift", &pitch_shift_js);
    function("trim", &trim_js);
    function("split", &split_js);
    function("preemphasis", &preemphasis_js);
    function("deemphasis", &deemphasis_js);
    function("remix", &remix_js);
    function("harmonicEffect", &harmonic_effect_js);
    function("percussiveEffect", &percussive_effect_js);

    function("medianFilter2d", &median_filter_2d_js);
    function("hpss", &hpss_js);
    function("decomposeNmf", &decompose_nmf_js);

    function("transitionUniform", &transition_uniform_js);
    function("transitionLoop", &transition_loop_js);
    function("transitionCycle", &transition_cycle_js);
    function("transitionLocal", &transition_local_js);
    function("viterbi", &viterbi_js);
    function("viterbiDiscriminative", &viterbi_discriminative_js);
    function("viterbiBinary", &viterbi_binary_js);
    function("cdistEuclidean", &cdist_euclidean_js);
    function("cdistCosine", &cdist_cosine_js);
    function("dtw", &dtw_js);
    function("rqa", &rqa_js);
    function("recurrenceToLag", &recurrence_to_lag_js);
    function("lagToRecurrence", &lag_to_recurrence_js);
    function("crossSimilarity", &cross_similarity_js);
    function("recurrenceMatrix", &recurrence_matrix_js);
    function("pathEnhance", &path_enhance_js);
    function("agglomerative", &agglomerative_js);
    function("subsegment", &subsegment_js);

    function("interpHarmonics", &interp_harmonics_js);
    function("f0Harmonics", &f0_harmonics_js);
    function("salience", &salience_js);

    function("frame", &frame_js);
    function("padCenter", &pad_center_js);
    function("fixLength", &fix_length_js);
    function("fixFrames", &fix_frames_js);
    function("normalize", &normalize_js);
    function("localmax", &localmax_js);
    function("localmin", &localmin_js);
    function("peakPick", &peak_pick_js);
    function("softmask", &softmask_js);
    function("abs2", &abs2_js);
    function("phasor", &phasor_js);
    function("fillOffDiagonal", &fill_off_diagonal_js);
    function("stack", &stack_js);
    function("validAudio", &valid_audio_js);
    function("isPositiveInt", &is_positive_int_js);
    function("validInt", &valid_int_js);

    function("pythagoreanIntervals", &pythagorean_intervals_js);
    function("plimitIntervals", &plimit_intervals_js);
    function("intervalFrequencies", &interval_frequencies_js);
    function("thaatToDegrees", &thaat_to_degrees_js);
    function("melaToDegrees", &mela_to_degrees_js);
    function("melaToSvara", &mela_to_svara_js);
    function("listMela", &list_mela_js);
    function("listThaat", &list_thaat_js);
    function("keyToDegrees", &key_to_degrees_js);
    function("keyToNotes", &key_to_notes_js);
    function("fifthsToNote", &fifths_to_note_js);
    function("intervalToFjs", &interval_to_fjs_js);
    function("midiToSvaraH", &midi_to_svara_h_js);
    function("midiToSvaraC", &midi_to_svara_c_js);
    function("hzToFjs", &hz_to_fjs_js);
}
