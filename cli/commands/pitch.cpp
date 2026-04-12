#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/core/pitch.hpp>
#include <memory>

namespace cli {

void register_pitch(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("pitch", "Estimate fundamental frequency (pYIN/YIN)");

    auto method = std::make_shared<std::string>("pyin");
    auto fmin = std::make_shared<double>(65.0);
    auto fmax = std::make_shared<double>(2093.0);
    sub->add_option("--method", *method, "Pitch method: pyin or yin")->default_val("pyin");
    sub->add_option("--fmin", *fmin, "Minimum frequency (Hz)")->default_val(65.0);
    sub->add_option("--fmax", *fmax, "Maximum frequency (Hz)")->default_val(2093.0);

    sub->callback([&opts, method, fmin, fmax]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "pitch", opts.input_file);

        if (*method == "pyin") {
            auto result = librosa::pyin(y, *fmin, *fmax, sr, opts.n_fft,
                                        std::optional<int>(opts.hop_length));
            auto times = make_time_axis(result.f0.size(), sr, opts.hop_length);
            out.pitch_output(times, result.f0, result.voiced_prob);
        } else {
            auto f0 = librosa::yin(y, *fmin, *fmax, sr, opts.n_fft,
                                   std::optional<int>(opts.hop_length));
            auto times = make_time_axis(f0.size(), sr, opts.hop_length);
            // YIN doesn't provide voiced probability, use 1.0 for all
            librosa::ArrayXr vp = librosa::ArrayXr::Ones(f0.size());
            out.pitch_output(times, f0, vp);
        }
    });
}

} // namespace cli
