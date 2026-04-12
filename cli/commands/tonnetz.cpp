#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/feature/spectral.hpp>

namespace cli {

void register_tonnetz(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("tonnetz", "Compute tonal centroid features");
    sub->callback([&]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto result = librosa::feature::tonnetz(y, sr);
        // shape: (6, n_frames)
        auto times = make_time_axis(result.cols(), sr, opts.hop_length);

        std::vector<std::string> labels = {"fifth_x", "fifth_y", "minor_x", "minor_y", "major_x", "major_y"};

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "tonnetz", opts.input_file);
        out.matrix(times, result, labels);
    });
}

} // namespace cli
