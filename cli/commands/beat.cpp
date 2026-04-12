#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/beat.hpp>
#include <memory>

namespace cli {

void register_beat(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("beat", "Detect beat positions");

    auto start_bpm = std::make_shared<double>(120.0);
    auto tightness = std::make_shared<double>(100.0);
    sub->add_option("--start-bpm", *start_bpm, "Initial tempo guess")->default_val(120.0);
    sub->add_option("--tightness", *tightness, "Beat tracking tightness")->default_val(100.0);

    sub->callback([&opts, start_bpm, tightness]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();

        auto [bpm, times] = librosa::beat::beat_track_times(y, audio.sample_rate,
                                                             opts.hop_length, *start_bpm, *tightness);

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "beat", opts.input_file);
        out.vector(times, "times");
    });
}

} // namespace cli
