#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/effects.hpp>
#include <sstream>
#include <memory>

namespace cli {

void register_trim(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("trim", "Find non-silent region boundaries");

    auto top_db = std::make_shared<double>(60.0);
    sub->add_option("--top-db", *top_db, "Threshold in dB below reference")->default_val(60.0);

    sub->callback([&opts, top_db]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto [trimmed, bounds] = librosa::effects::trim(y, *top_db, -1,
                                                         opts.n_fft, opts.hop_length);
        auto [start_idx, end_idx] = bounds;

        double start_time = static_cast<double>(start_idx) / sr;
        double end_time = static_cast<double>(end_idx) / sr;

        std::ostringstream ss_start, ss_end, ss_si, ss_ei;
        ss_start << std::fixed << std::setprecision(opts.precision) << start_time;
        ss_end << std::fixed << std::setprecision(opts.precision) << end_time;
        ss_si << start_idx;
        ss_ei << end_idx;

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "trim", opts.input_file);
        out.key_value({
            {"start_time", ss_start.str()},
            {"end_time", ss_end.str()},
            {"start_sample", ss_si.str()},
            {"end_sample", ss_ei.str()}
        });
    });
}

} // namespace cli
