#include "common.hpp"
#include <librosa/util/exceptions.hpp>
#include <stdexcept>

namespace cli {

librosa::AudioData load_audio(const CommonOptions& opts) {
    std::optional<librosa::Real> sr_opt;
    if (opts.sr > 0) {
        sr_opt = opts.sr;
    } else {
        sr_opt = std::nullopt; // native rate
    }

    std::optional<librosa::Real> dur_opt;
    if (opts.duration >= 0) {
        dur_opt = opts.duration;
    } else if (opts.duration != -1.0) {
        throw librosa::ParameterError("duration must be non-negative or -1");
    }

    return librosa::load(opts.input_file, sr_opt, opts.mono, opts.offset, dur_opt);
}

librosa::ArrayXr make_time_axis(int n_frames, double sr, int hop_length) {
    librosa::ArrayXr frames = librosa::ArrayXr::LinSpaced(n_frames, 0, n_frames - 1);
    return librosa::frames_to_time(frames, sr, hop_length);
}

librosa::WindowType parse_window(const std::string& name) {
    if (name == "hann") return librosa::WindowType::Hann;
    if (name == "hamming") return librosa::WindowType::Hamming;
    if (name == "blackman") return librosa::WindowType::Blackman;
    if (name == "bartlett") return librosa::WindowType::Bartlett;
    if (name == "rectangular" || name == "rect" || name == "boxcar")
        return librosa::WindowType::Rectangular;
    throw std::runtime_error("Unknown window type: " + name);
}

} // namespace cli
