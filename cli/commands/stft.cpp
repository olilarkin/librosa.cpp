#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/core/spectrum.hpp>

namespace cli {

void register_stft(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("stft", "Compute magnitude spectrogram (in dB)");
    sub->callback([&]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        auto S = librosa::stft(y, opts.n_fft, opts.hop_length,
                                std::optional<int>(opts.effective_win_length()),
                                parse_window(opts.window));
        auto mag = librosa::magnitude(S);
        auto mag_db = librosa::amplitude_to_db(mag);
        auto times = make_time_axis(mag_db.cols(), sr, opts.hop_length);

        int n_bins = opts.n_fft / 2 + 1;
        std::vector<std::string> labels;
        for (int i = 0; i < n_bins; ++i)
            labels.push_back("f" + std::to_string(i));

        OutputFormatter out(opts.format, opts.precision, opts.no_time, "stft", opts.input_file);
        out.matrix(times, mag_db, labels);
    });
}

} // namespace cli
