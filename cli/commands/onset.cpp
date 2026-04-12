#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/onset.hpp>
#include <memory>

namespace cli {

void register_onset(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("onset", "Detect onset positions");

    auto backtrack = std::make_shared<bool>(false);
    sub->add_flag("--backtrack", *backtrack, "Backtrack onsets to nearest minimum");

    sub->callback([&opts, backtrack]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();

        auto times = librosa::onset::onset_detect_times(y, audio.sample_rate,
                                                         opts.hop_length, *backtrack);

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "onset", opts.input_file);
        out.vector(times, "times");
    });
}

} // namespace cli
