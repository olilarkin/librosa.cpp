#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/core/pitch.hpp>

namespace cli {

void register_tuning(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("tuning", "Estimate tuning offset (cents)");
    sub->callback([&]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto tuning = librosa::estimate_tuning(y, sr, opts.n_fft);

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "tuning", opts.input_file);
        out.scalar("tuning", tuning);
    });
}

} // namespace cli
