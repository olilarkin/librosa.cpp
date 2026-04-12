#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/core/audio.hpp>
#include <iomanip>
#include <sstream>

namespace cli {

void register_info(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("info", "Show audio file metadata");
    sub->callback([&]() {
        OutputFormatter out(opts.format, opts.precision, opts.no_time, "info", opts.input_file);

        auto info = librosa::get_audio_info(opts.input_file);

        std::ostringstream sr_ss, dur_ss, samp_ss, ch_ss;
        sr_ss << info.sample_rate;
        dur_ss << std::fixed << std::setprecision(opts.precision) << info.duration;
        samp_ss << info.samples;
        ch_ss << info.channels;

        out.key_value({
            {"filename", opts.input_file},
            {"native_sr", sr_ss.str()},
            {"duration", dur_ss.str()},
            {"samples", samp_ss.str()},
            {"channels", ch_ss.str()}
        });
    });
}

} // namespace cli
