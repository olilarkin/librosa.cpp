#include "benchmark_helpers.hpp"

#include <librosa/beat.hpp>
#include <librosa/core/constantq.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/spectrum.hpp>
#include <librosa/feature/inverse.hpp>
#include <librosa/feature/rhythm.hpp>
#include <librosa/feature/spectral.hpp>
#include <librosa/feature/utils.hpp>

#include <limits>
#include <optional>
#include <vector>

namespace {

using namespace librosa_bench;

using MelSpecFn = librosa::ArrayXXr (*)(
    const librosa::ArrayXXr&, librosa::Real, int, int, librosa::Real,
    std::optional<librosa::Real>, bool, bool);
using ChromaStftAudioFn = librosa::ArrayXXr (*)(
    const librosa::ArrayXr&, librosa::Real, int, int, int,
    std::optional<librosa::Real>, librosa::Real, librosa::WindowType, bool);
using ChromaStftSpecFn = librosa::ArrayXXr (*)(
    const librosa::ArrayXXr&, librosa::Real, int, int,
    std::optional<librosa::Real>, librosa::Real);
using FlatnessSpecFn = librosa::ArrayXXr (*)(const librosa::ArrayXXr&,
                                             librosa::Real, librosa::Real);
using RMSSpecFn = librosa::ArrayXXr (*)(const librosa::ArrayXXr&, int);

void BM_FeatureMelAndMFCC(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));
    const auto D = librosa::stft(y, 1024, 256);
    const librosa::ArrayXXr S = librosa::magnitude(D).square();

    for (auto _ : state) {
        auto mel_audio = librosa::feature::melspectrogram(
            y, kSampleRate, 1024, 256, std::nullopt, librosa::WindowType::Hann,
            true, librosa::PadMode::Constant, 2.0, 64);
        auto mel_spec = static_cast<MelSpecFn>(&librosa::feature::melspectrogram)(
            S, kSampleRate, 1024, 64, 0.0, std::nullopt, false, true);
        auto mfcc_audio = librosa::feature::mfcc(y, kSampleRate, 20, 2, true, 0.0,
                                                 1024, 256, 64);
        auto mfcc_spec = librosa::feature::mfcc(librosa::power_to_db(mel_spec), 20);
        consume_eigen(mel_audio);
        consume_eigen(mel_spec);
        consume_eigen(mfcc_audio);
        consume_eigen(mfcc_spec);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_FeatureChroma(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));
    const auto D = librosa::stft(y, 2048, kHopLength);
    const librosa::ArrayXXr S = librosa::magnitude(D).square();
    const auto C = librosa::cqt(y, kSampleRate, kHopLength, std::nullopt, 36, 12, 0.0);
    const auto V = librosa::vqt(y, kSampleRate, kHopLength, std::nullopt, 36,
                                0.0, 12, 0.0);

    for (auto _ : state) {
        auto stft_audio = static_cast<ChromaStftAudioFn>(&librosa::feature::chroma_stft)(
            y, kSampleRate, 2048, kHopLength, 12, 0.0,
            std::numeric_limits<librosa::Real>::infinity(),
            librosa::WindowType::Hann, true);
        auto stft_spec = static_cast<ChromaStftSpecFn>(&librosa::feature::chroma_stft)(
            S, kSampleRate, 2048, 12, 0.0,
            std::numeric_limits<librosa::Real>::infinity());
        auto cqt_audio = librosa::feature::chroma_cqt(
            y, kSampleRate, kHopLength, std::nullopt,
            std::numeric_limits<librosa::Real>::infinity(), 0.0, 0.0, 12, 3, 12);
        auto cqt_spec = librosa::feature::chroma_cqt(
            C.abs(), std::nullopt, std::numeric_limits<librosa::Real>::infinity(),
            0.0, 12, 12);
        auto cens = librosa::feature::chroma_cens(
            y, kSampleRate, kHopLength, std::nullopt, 0.0, 12, 3, 12, 2.0, 9);
        auto vqt_audio = librosa::feature::chroma_vqt(
            y, kSampleRate, kHopLength, std::nullopt,
            std::numeric_limits<librosa::Real>::infinity(), 0.0, 3, 12, 0.0);
        auto vqt_spec = librosa::feature::chroma_vqt(
            V.abs(), std::nullopt, std::numeric_limits<librosa::Real>::infinity(),
            0.0, 12);

        consume_eigen(stft_audio);
        consume_eigen(stft_spec);
        consume_eigen(cqt_audio);
        consume_eigen(cqt_spec);
        consume_eigen(cens);
        consume_eigen(vqt_audio);
        consume_eigen(vqt_spec);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_FeatureSpectralShape(benchmark::State& state) {
    const librosa::ArrayXr y =
        (make_tone(440.0, state.range(0)) + 0.25 * make_tone(1760.0, state.range(0))).eval();
    const auto D = librosa::stft(y, 1024, 256);
    const librosa::ArrayXXr S = (librosa::magnitude(D) + 1e-6).eval();
    const auto freqs = librosa::fft_frequencies(kSampleRate, 1024);

    for (auto _ : state) {
        auto centroid_audio = librosa::feature::spectral_centroid(y, kSampleRate, 1024, 256);
        auto centroid_spec = librosa::feature::spectral_centroid(S, kSampleRate, 1024, &freqs);
        auto bandwidth_audio = librosa::feature::spectral_bandwidth(y, kSampleRate, 1024, 256);
        auto bandwidth_spec = librosa::feature::spectral_bandwidth(
            S, kSampleRate, 1024, &centroid_spec, &freqs);
        auto rolloff_audio = librosa::feature::spectral_rolloff(y, kSampleRate, 1024, 256);
        auto rolloff_spec = librosa::feature::spectral_rolloff(S, kSampleRate, 1024, &freqs);
        auto flatness_audio = librosa::feature::spectral_flatness(y, 1024, 256);
        auto flatness_spec = static_cast<FlatnessSpecFn>(&librosa::feature::spectral_flatness)(
            S, 1e-10, 2.0);
        auto contrast_audio = librosa::feature::spectral_contrast(y, kSampleRate, 1024, 256);
        auto contrast_spec = librosa::feature::spectral_contrast(S, kSampleRate, 1024, &freqs);

        consume_eigen(centroid_audio);
        consume_eigen(centroid_spec);
        consume_eigen(bandwidth_audio);
        consume_eigen(bandwidth_spec);
        consume_eigen(rolloff_audio);
        consume_eigen(rolloff_spec);
        consume_eigen(flatness_audio);
        consume_eigen(flatness_spec);
        consume_eigen(contrast_audio);
        consume_eigen(contrast_spec);
    }

    state.SetItemsProcessed(state.iterations() * S.size());
}

void BM_FeatureEnergyPolynomialTonnetz(benchmark::State& state) {
    const librosa::ArrayXr y =
        (make_tone(440.0, state.range(0)) + 0.3 * make_tone(660.0, state.range(0))).eval();
    const auto D = librosa::stft(y, 1024, 256);
    const librosa::ArrayXXr S = (librosa::magnitude(D) + 1e-6).eval();
    const auto freqs = librosa::fft_frequencies(kSampleRate, 1024);
    const auto chroma = static_cast<ChromaStftAudioFn>(&librosa::feature::chroma_stft)(
        y, kSampleRate, 2048, kHopLength, 12, 0.0,
        std::numeric_limits<librosa::Real>::infinity(),
        librosa::WindowType::Hann, true);

    for (auto _ : state) {
        auto rms_audio = librosa::feature::rms(y, 1024, 256);
        auto rms_spec = static_cast<RMSSpecFn>(&librosa::feature::rms)(S, 1024);
        auto zcr = librosa::feature::zero_crossing_rate(y, 1024, 256);
        auto poly_audio = librosa::feature::poly_features(y, kSampleRate, 1024, 256, 2);
        auto poly_spec = librosa::feature::poly_features(S, kSampleRate, 1024, 2, &freqs);
        auto tonnetz_audio = librosa::feature::tonnetz(y, kSampleRate, &chroma);
        auto tonnetz_chroma = librosa::feature::tonnetz(chroma);

        consume_eigen(rms_audio);
        consume_eigen(rms_spec);
        consume_eigen(zcr);
        consume_eigen(poly_audio);
        consume_eigen(poly_spec);
        consume_eigen(tonnetz_audio);
        consume_eigen(tonnetz_chroma);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_FeatureDeltaStackMemory(benchmark::State& state) {
    const auto data = make_feature_matrix(16, state.range(0));

    for (auto _ : state) {
        auto d1 = librosa::feature::delta(data, 9, 1, -1, "interp");
        auto d2 = librosa::feature::delta(data, 9, 2, -1, "interp");
        auto memory = librosa::feature::stack_memory(data, 4, 2);
        consume_eigen(d1);
        consume_eigen(d2);
        consume_eigen(memory);
    }

    state.SetItemsProcessed(state.iterations() * data.size());
}

void BM_FeatureInverse(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));
    const auto mel = librosa::feature::melspectrogram(
        y, kSampleRate, 1024, 256, std::nullopt, librosa::WindowType::Hann,
        true, librosa::PadMode::Constant, 2.0, 64);
    const auto mfcc = librosa::feature::mfcc(y, kSampleRate, 20, 2, true, 0.0,
                                             1024, 256, 64);

    for (auto _ : state) {
        auto stft = librosa::feature::mel_to_stft(mel, kSampleRate, 1024);
        auto audio = librosa::feature::mel_to_audio(mel, kSampleRate, 1024, 256,
                                                    std::nullopt, librosa::WindowType::Hann,
                                                    true, 2.0, 2,
                                                    static_cast<int>(y.size()));
        auto mel_from_mfcc = librosa::feature::mfcc_to_mel(mfcc, 64);
        auto audio_from_mfcc = librosa::feature::mfcc_to_audio(
            mfcc, 64, 2, true, 1.0, 0.0, kSampleRate, 1024, 256, 2);

        consume_eigen(stft);
        consume_eigen(audio);
        consume_eigen(mel_from_mfcc);
        consume_eigen(audio_from_mfcc);
    }

    state.SetItemsProcessed(state.iterations() * mel.size());
}

void BM_FeatureRhythm(benchmark::State& state) {
    const auto envelope = make_onset_envelope(state.range(0));
    const auto y = make_click_track(state.range(0) * kHopLength);
    const auto tg = librosa::beat::tempogram(envelope, kSampleRate, kHopLength, 128);
    const std::vector<librosa::Real> factors = {0.5, 1.0, 2.0, 3.0};

    for (auto _ : state) {
        auto fourier = librosa::feature::fourier_tempogram(
            envelope, kSampleRate, kHopLength, 128);
        auto fourier_audio = librosa::feature::fourier_tempogram_audio(
            y, kSampleRate, kHopLength, 128);
        auto ratio = librosa::feature::tempogram_ratio(
            tg, kSampleRate, kHopLength, factors, 120.0, 1.0, 320.0);

        consume_eigen(fourier);
        consume_eigen(fourier_audio);
        consume_eigen(ratio);
    }

    state.SetItemsProcessed(state.iterations() * envelope.size());
}

} // namespace

BENCHMARK(BM_FeatureMelAndMFCC)->Arg(8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FeatureChroma)->Arg(8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FeatureSpectralShape)->Arg(8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FeatureEnergyPolynomialTonnetz)->Arg(8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FeatureDeltaStackMemory)->Arg(512)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_FeatureInverse)->Arg(4096)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FeatureRhythm)->Arg(1024)->Unit(benchmark::kMillisecond);
