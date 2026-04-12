#pragma once

#include <librosa/types.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/core/convert.hpp>
#include <string>
#include <optional>

namespace cli {

struct CommonOptions {
    // Input
    std::string input_file;

    // Audio loading
    double sr = 22050;
    bool mono = true;
    double offset = 0.0;
    double duration = -1.0; // negative = load all

    // STFT parameters
    int n_fft = 2048;
    int hop_length = 512;
    int win_length = 0; // 0 = use n_fft
    std::string window = "hann";

    // Output
    std::string format = "text";
    int precision = 6;
    bool no_time = false;

    // Derived
    int effective_win_length() const { return win_length > 0 ? win_length : n_fft; }
};

/// Load audio using CommonOptions settings
librosa::AudioData load_audio(const CommonOptions& opts);

/// Build a time axis for n_frames frames
librosa::ArrayXr make_time_axis(int n_frames, double sr, int hop_length);

/// Parse window string to WindowType
librosa::WindowType parse_window(const std::string& name);

} // namespace cli
