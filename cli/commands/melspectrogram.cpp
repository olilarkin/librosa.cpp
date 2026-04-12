#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/feature/spectral.hpp>
#include <librosa/core/spectrum.hpp>
#include <memory>

namespace cli {

void register_melspectrogram(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("melspectrogram", "Compute mel spectrogram (in dB)");

    auto n_mels = std::make_shared<int>(128);
    sub->add_option("--n-mels", *n_mels, "Number of mel bands")->default_val(128);

    sub->callback([&opts, n_mels]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto mel = librosa::feature::melspectrogram(
            y, sr, opts.n_fft, opts.hop_length, std::nullopt,
            parse_window(opts.window), true, librosa::PadMode::Constant,
            2.0, *n_mels);

        auto mel_db = librosa::power_to_db(mel);
        auto times = make_time_axis(mel_db.cols(), sr, opts.hop_length);

        std::vector<std::string> labels;
        for (int i = 0; i < *n_mels; ++i)
            labels.push_back("m" + std::to_string(i));

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "melspectrogram", opts.input_file);
        out.matrix(times, mel_db, labels);
    });
}

} // namespace cli
