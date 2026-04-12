#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/feature/spectral.hpp>
#include <memory>

namespace cli {

void register_spectral_centroid(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("spectral-centroid", "Compute spectral centroid");
    sub->callback([&opts]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto result = librosa::feature::spectral_centroid(y, sr, opts.n_fft, opts.hop_length,
                                                           parse_window(opts.window));
        auto times = make_time_axis(result.cols(), sr, opts.hop_length);

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "spectral-centroid", opts.input_file);
        out.per_frame(times, result, {"centroid"});
    });
}

void register_spectral_bandwidth(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("spectral-bandwidth", "Compute spectral bandwidth");
    sub->callback([&opts]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto result = librosa::feature::spectral_bandwidth(y, sr, opts.n_fft, opts.hop_length,
                                                            parse_window(opts.window));
        auto times = make_time_axis(result.cols(), sr, opts.hop_length);

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "spectral-bandwidth", opts.input_file);
        out.per_frame(times, result, {"bandwidth"});
    });
}

void register_spectral_rolloff(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("spectral-rolloff", "Compute spectral rolloff");

    auto roll_percent = std::make_shared<double>(0.85);
    sub->add_option("--roll-percent", *roll_percent, "Roll-off percentage")->default_val(0.85);

    sub->callback([&opts, roll_percent]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto result = librosa::feature::spectral_rolloff(y, sr, opts.n_fft, opts.hop_length,
                                                          parse_window(opts.window), true, *roll_percent);
        auto times = make_time_axis(result.cols(), sr, opts.hop_length);

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "spectral-rolloff", opts.input_file);
        out.per_frame(times, result, {"rolloff"});
    });
}

void register_spectral_flatness(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("spectral-flatness", "Compute spectral flatness");
    sub->callback([&opts]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();

        auto result = librosa::feature::spectral_flatness(y, opts.n_fft, opts.hop_length,
                                                           parse_window(opts.window));
        auto times = make_time_axis(result.cols(), audio.sample_rate, opts.hop_length);

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "spectral-flatness", opts.input_file);
        out.per_frame(times, result, {"flatness"});
    });
}

void register_spectral_contrast(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("spectral-contrast", "Compute spectral contrast");

    auto n_bands = std::make_shared<int>(6);
    sub->add_option("--n-bands", *n_bands, "Number of frequency bands")->default_val(6);

    sub->callback([&opts, n_bands]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto result = librosa::feature::spectral_contrast(y, sr, opts.n_fft, opts.hop_length,
                                                           parse_window(opts.window), true, 200.0, *n_bands);
        auto times = make_time_axis(result.cols(), sr, opts.hop_length);

        std::vector<std::string> labels;
        for (int i = 0; i < *n_bands; ++i)
            labels.push_back("band" + std::to_string(i));
        labels.push_back("valley");

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "spectral-contrast", opts.input_file);
        out.matrix(times, result, labels);
    });
}

} // namespace cli
