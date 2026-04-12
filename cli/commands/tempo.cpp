#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/beat.hpp>
#include <memory>

namespace cli {

void register_tempo(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("tempo", "Estimate tempo in BPM");

    auto start_bpm = std::make_shared<double>(120.0);
    sub->add_option("--start-bpm", *start_bpm, "Initial tempo guess")->default_val(120.0);

    sub->callback([&opts, start_bpm]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();

        auto bpm = librosa::beat::tempo_audio(y, audio.sample_rate, opts.hop_length, *start_bpm);

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "tempo", opts.input_file);
        out.scalar("bpm", bpm);
    });
}

} // namespace cli
