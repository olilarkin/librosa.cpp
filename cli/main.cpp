#include <CLI/CLI.hpp>
#include "common.hpp"
#include <iostream>

namespace cli {
// Forward declarations for all register_* functions
void register_info(CLI::App& app, CommonOptions& opts);
void register_tempo(CLI::App& app, CommonOptions& opts);
void register_beat(CLI::App& app, CommonOptions& opts);
void register_onset(CLI::App& app, CommonOptions& opts);
void register_pitch(CLI::App& app, CommonOptions& opts);
void register_mfcc(CLI::App& app, CommonOptions& opts);
void register_melspectrogram(CLI::App& app, CommonOptions& opts);
void register_chroma(CLI::App& app, CommonOptions& opts);
void register_spectral_centroid(CLI::App& app, CommonOptions& opts);
void register_spectral_bandwidth(CLI::App& app, CommonOptions& opts);
void register_spectral_rolloff(CLI::App& app, CommonOptions& opts);
void register_spectral_flatness(CLI::App& app, CommonOptions& opts);
void register_spectral_contrast(CLI::App& app, CommonOptions& opts);
void register_rms(CLI::App& app, CommonOptions& opts);
void register_zcr(CLI::App& app, CommonOptions& opts);
void register_tonnetz(CLI::App& app, CommonOptions& opts);
void register_tuning(CLI::App& app, CommonOptions& opts);
void register_stft(CLI::App& app, CommonOptions& opts);
void register_trim(CLI::App& app, CommonOptions& opts);
void register_hpss(CLI::App& app, CommonOptions& opts);
} // namespace cli

int main(int argc, char** argv) {
    CLI::App app{"rosa: Audio analysis from the command line"};
    app.require_subcommand(1);

    cli::CommonOptions opts;

    // Global flags — input file
    app.add_option("file", opts.input_file, "Audio file path")->required()->check(CLI::ExistingFile);

    // Audio loading options
    app.add_option("--sr", opts.sr, "Sample rate (0 = native)")->default_val(22050);
    app.add_flag("--mono,!--no-mono", opts.mono, "Force mono")->default_val(true);
    app.add_option("--offset", opts.offset, "Start offset (seconds)")->default_val(0.0);
    app.add_option("--duration", opts.duration, "Duration to load (seconds, -1 = all)")->default_val(-1.0);

    // STFT options
    app.add_option("--n-fft", opts.n_fft, "FFT size")->default_val(2048);
    app.add_option("--hop-length", opts.hop_length, "Hop length")->default_val(512);
    app.add_option("--win-length", opts.win_length, "Window length (0 = n_fft)")->default_val(0);
    app.add_option("--window", opts.window, "Window type: hann|hamming|blackman|bartlett")->default_val("hann");

    // Output options
    app.add_option("--format", opts.format, "Output format: text|json|csv")->default_val("text");
    app.add_option("--precision", opts.precision, "Decimal places")->default_val(6);
    app.add_flag("--no-time", opts.no_time, "Omit time column");

    // Register all subcommands
    cli::register_info(app, opts);
    cli::register_tempo(app, opts);
    cli::register_beat(app, opts);
    cli::register_onset(app, opts);
    cli::register_pitch(app, opts);
    cli::register_mfcc(app, opts);
    cli::register_melspectrogram(app, opts);
    cli::register_chroma(app, opts);
    cli::register_spectral_centroid(app, opts);
    cli::register_spectral_bandwidth(app, opts);
    cli::register_spectral_rolloff(app, opts);
    cli::register_spectral_flatness(app, opts);
    cli::register_spectral_contrast(app, opts);
    cli::register_rms(app, opts);
    cli::register_zcr(app, opts);
    cli::register_tonnetz(app, opts);
    cli::register_tuning(app, opts);
    cli::register_stft(app, opts);
    cli::register_trim(app, opts);
    cli::register_hpss(app, opts);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
