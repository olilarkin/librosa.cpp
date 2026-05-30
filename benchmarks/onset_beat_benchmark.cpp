#include "benchmark_helpers.hpp"

#include <librosa/beat.hpp>
#include <librosa/onset.hpp>

#include <optional>
#include <vector>

namespace {

using namespace librosa_bench;

void BM_OnsetMaximumFilterAndMatching(benchmark::State& state) {
    const auto S = make_positive_matrix(128, state.range(0));
    std::vector<Eigen::Index> from;
    std::vector<Eigen::Index> to;
    for (Eigen::Index i = 0; i < state.range(0); i += 7) {
        from.push_back(i);
    }
    for (Eigen::Index i = 0; i < state.range(0); i += 5) {
        to.push_back(i);
    }

    for (auto _ : state) {
        auto filtered_freq = librosa::onset::maximum_filter1d(S, 5, -2);
        auto filtered_time = librosa::onset::maximum_filter1d(S, 5, -1);
        auto matches = librosa::onset::match_events(from, to);
        consume_eigen(filtered_freq);
        consume_eigen(filtered_time);
        consume_vector(matches);
    }

    state.SetItemsProcessed(state.iterations() * S.size());
}

void BM_OnsetStrengthMulti(benchmark::State& state) {
    const auto y = make_click_track(state.range(0));
    const auto S = make_positive_matrix(128, state.range(0) / kHopLength + 1);
    const std::vector<int> channels = {0, 32, 64, 96, 128};

    for (auto _ : state) {
        auto single_audio = librosa::onset::onset_strength(y, kSampleRate, kNFFT, kHopLength);
        auto single_spec = librosa::onset::onset_strength(S, kSampleRate, kNFFT, kHopLength);
        auto multi_audio = librosa::onset::onset_strength_multi(
            y, kSampleRate, kNFFT, kHopLength, 1, 1, false, true, channels);
        auto multi_spec = librosa::onset::onset_strength_multi(
            S, kSampleRate, kNFFT, kHopLength, 1, 1, false, true, channels);
        consume_eigen(single_audio);
        consume_eigen(single_spec);
        consume_eigen(multi_audio);
        consume_eigen(multi_spec);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_OnsetDetection(benchmark::State& state) {
    const auto y = make_click_track(state.range(0));
    const auto envelope = make_onset_envelope(state.range(0) / kHopLength + 1);

    for (auto _ : state) {
        auto from_audio = librosa::onset::onset_detect(
            y, kSampleRate, kHopLength, false, librosa::onset::OnsetUnits::Frames,
            true, 3, 3, 3, 3, 0.1, 3);
        auto from_envelope = librosa::onset::onset_detect_envelope(
            envelope, kSampleRate, kHopLength, false, librosa::onset::OnsetUnits::Frames,
            true, 3, 3, 3, 3, 0.1, 3);
        auto times = librosa::onset::onset_detect_times(y, kSampleRate, kHopLength);
        auto backtracked = librosa::onset::onset_backtrack(from_envelope, envelope);
        consume_vector(from_audio);
        consume_vector(from_envelope);
        consume_eigen(times);
        consume_vector(backtracked);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_BeatTempogramAndTempo(benchmark::State& state) {
    const auto envelope = make_onset_envelope(state.range(0));
    const auto y = make_click_track(state.range(0) * kHopLength);

    for (auto _ : state) {
        auto tg = librosa::beat::tempogram(envelope, kSampleRate, kHopLength, 128);
        auto tg_audio = librosa::beat::tempogram_audio(y, kSampleRate, kHopLength, 128);
        auto tempo = librosa::beat::tempo(envelope, kSampleRate, kHopLength);
        auto tempo_audio = librosa::beat::tempo_audio(y, kSampleRate, kHopLength);
        auto tempo_frames = librosa::beat::tempo_frames(envelope, kSampleRate, kHopLength);
        benchmark::DoNotOptimize(tempo);
        benchmark::DoNotOptimize(tempo_audio);
        consume_eigen(tg);
        consume_eigen(tg_audio);
        consume_eigen(tempo_frames);
    }

    state.SetItemsProcessed(state.iterations() * envelope.size());
}

void BM_BeatTrackVariants(benchmark::State& state) {
    const auto envelope = make_onset_envelope(state.range(0));
    const auto y = make_click_track(state.range(0) * kHopLength);

    for (auto _ : state) {
        auto beats = librosa::beat::beat_track(envelope, kSampleRate, kHopLength);
        auto beat_audio = librosa::beat::beat_track_audio(y, kSampleRate, kHopLength);
        auto beat_times = librosa::beat::beat_track_times(y, kSampleRate, kHopLength);
        benchmark::DoNotOptimize(beats.first);
        consume_vector(beats.second);
        benchmark::DoNotOptimize(beat_audio.first);
        consume_vector(beat_audio.second);
        benchmark::DoNotOptimize(beat_times.first);
        consume_eigen(beat_times.second);
    }

    state.SetItemsProcessed(state.iterations() * envelope.size());
}

void BM_BeatPLP(benchmark::State& state) {
    const auto envelope = make_onset_envelope(state.range(0));
    const auto y = make_click_track(state.range(0) * kHopLength);

    for (auto _ : state) {
        auto plp = librosa::beat::plp(envelope, kSampleRate, kHopLength, 128);
        auto plp_audio = librosa::beat::plp_audio(y, kSampleRate, kHopLength, 128);
        consume_eigen(plp);
        consume_eigen(plp_audio);
    }

    state.SetItemsProcessed(state.iterations() * envelope.size());
}

} // namespace

BENCHMARK(BM_OnsetMaximumFilterAndMatching)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_OnsetStrengthMulti)->Arg(22050)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_OnsetDetection)->Arg(22050)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BeatTempogramAndTempo)->Arg(1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BeatTrackVariants)->Arg(1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_BeatPLP)->Arg(1024)->Unit(benchmark::kMillisecond);
