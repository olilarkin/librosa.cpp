#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/feature/spectral.hpp>

namespace cli {

void register_rms(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("rms", "Compute root-mean-square energy");
    sub->callback([&]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto result = librosa::feature::rms(y, opts.n_fft, opts.hop_length);
        auto times = make_time_axis(result.cols(), sr, opts.hop_length);

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "rms", opts.input_file);
        out.per_frame(times, result, {"rms"});
    });
}

} // namespace cli
