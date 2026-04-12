#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/feature/spectral.hpp>
#include <memory>

namespace cli {

void register_chroma(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("chroma", "Compute chromagram (stft/cqt/cens)");

    auto variant = std::make_shared<std::string>("stft");
    sub->add_option("--variant", *variant, "Chroma variant: stft, cqt, cens")->default_val("stft");

    sub->callback([&opts, variant]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        librosa::ArrayXXr result;
        if (*variant == "cqt") {
            result = librosa::feature::chroma_cqt(y, sr, opts.hop_length);
        } else if (*variant == "cens") {
            result = librosa::feature::chroma_cens(y, sr, opts.hop_length);
        } else {
            result = librosa::feature::chroma_stft(y, sr, opts.n_fft, opts.hop_length);
        }

        // result shape: (12, n_frames)
        auto times = make_time_axis(result.cols(), sr, opts.hop_length);

        std::vector<std::string> labels = {"C", "C#", "D", "D#", "E", "F",
                                            "F#", "G", "G#", "A", "A#", "B"};

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "chroma", opts.input_file);
        out.matrix(times, result, labels);
    });
}

} // namespace cli
