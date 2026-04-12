#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/feature/spectral.hpp>
#include <memory>

namespace cli {

void register_mfcc(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("mfcc", "Compute Mel-frequency cepstral coefficients");

    auto n_mfcc = std::make_shared<int>(20);
    auto n_mels = std::make_shared<int>(128);
    sub->add_option("--n-mfcc", *n_mfcc, "Number of MFCCs")->default_val(20);
    sub->add_option("--n-mels", *n_mels, "Number of mel bands")->default_val(128);

    sub->callback([&opts, n_mfcc, n_mels]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto result = librosa::feature::mfcc(y, sr, *n_mfcc, 2, true, 0,
                                              opts.n_fft, opts.hop_length, *n_mels);
        // result shape: (n_mfcc, n_frames)
        auto times = make_time_axis(result.cols(), sr, opts.hop_length);

        std::vector<std::string> labels;
        for (int i = 0; i < *n_mfcc; ++i)
            labels.push_back("c" + std::to_string(i));

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "mfcc", opts.input_file);
        out.matrix(times, result, labels);
    });
}

} // namespace cli
