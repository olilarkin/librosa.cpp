#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/feature/spectral.hpp>

namespace cli {

void register_zcr(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("zcr", "Compute zero crossing rate");
    sub->callback([&]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto result = librosa::feature::zero_crossing_rate(y, opts.n_fft, opts.hop_length);
        auto times = make_time_axis(result.cols(), sr, opts.hop_length);

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "zcr", opts.input_file);
        out.per_frame(times, result, {"zcr"});
    });
}

} // namespace cli
