#include <benchmark/benchmark.h>
#include <librosa/beat.hpp>
#include <librosa/core/audio.hpp>
#include <librosa/onset.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

constexpr librosa::Real kSampleRate = 22050.0;
constexpr int kHopLength = 512;
constexpr int kNFFT = 2048;

librosa::ArrayXr make_onset_envelope(Eigen::Index n_frames, int period = 22) {
    librosa::ArrayXr envelope(n_frames);

    for (Eigen::Index i = 0; i < n_frames; ++i) {
        const bool beat = (i % period) == 0;
        const librosa::Real slow_mod =
            0.1 * std::sin(2.0 * librosa::constants::PI * static_cast<librosa::Real>(i) / 97.0);
        envelope(i) = beat ? 1.0 + slow_mod : 0.03;
    }

    return envelope;
}

librosa::ArrayXr make_click_track(int seconds, int period_frames = 22) {
    const Eigen::Index n_samples = static_cast<Eigen::Index>(seconds * kSampleRate);
    librosa::ArrayXr y = librosa::ArrayXr::Zero(n_samples);
    const Eigen::Index hop = kHopLength;
    const Eigen::Index n_frames = n_samples / hop;

    for (Eigen::Index frame = 0; frame < n_frames; frame += period_frames) {
        const Eigen::Index sample = frame * hop;
        if (sample < n_samples) {
            y(sample) = 1.0;
        }
    }

    return y;
}

const librosa::ArrayXr& real_audio() {
    static const librosa::ArrayXr audio = [] {
        const char* path = std::getenv("LIBROSA_BENCH_AUDIO");
        if (path == nullptr || std::string(path).empty()) {
            throw std::runtime_error("set LIBROSA_BENCH_AUDIO to enable real-audio benchmarks");
        }

        auto data = librosa::load(path, kSampleRate, true);
        return data.mono();
    }();

    return audio;
}

bool has_real_audio_path() {
    const char* path = std::getenv("LIBROSA_BENCH_AUDIO");
    return path != nullptr && !std::string(path).empty();
}

const char* real_audio_path() {
    const char* path = std::getenv("LIBROSA_BENCH_AUDIO");
    if (path == nullptr || std::string(path).empty()) {
        throw std::runtime_error("set LIBROSA_BENCH_AUDIO to enable real-audio benchmarks");
    }
    return path;
}

void BM_OnsetStrengthSyntheticAudio(benchmark::State& state) {
    const auto y = make_click_track(static_cast<int>(state.range(0)));

    for (auto _ : state) {
        auto envelope = librosa::onset::onset_strength(y, kSampleRate, kNFFT, kHopLength);
        benchmark::DoNotOptimize(envelope.data());
        benchmark::DoNotOptimize(envelope.size());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_TempoEnvelope(benchmark::State& state) {
    const auto envelope = make_onset_envelope(state.range(0));

    for (auto _ : state) {
        auto bpm = librosa::beat::tempo(envelope, kSampleRate, kHopLength);
        benchmark::DoNotOptimize(bpm);
    }

    state.SetItemsProcessed(state.iterations() * envelope.size());
}

void BM_BeatTrackFixedTempoEnvelope(benchmark::State& state) {
    const auto envelope = make_onset_envelope(state.range(0));

    for (auto _ : state) {
        auto result = librosa::beat::beat_track(
            envelope, kSampleRate, kHopLength, 120.0, 100.0, true, 120.0);
        benchmark::DoNotOptimize(result.first);
        benchmark::DoNotOptimize(result.second.data());
        benchmark::DoNotOptimize(result.second.size());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * envelope.size());
}

void BM_BeatTrackEstimatedTempoEnvelope(benchmark::State& state) {
    const auto envelope = make_onset_envelope(state.range(0));

    for (auto _ : state) {
        auto result = librosa::beat::beat_track(envelope, kSampleRate, kHopLength);
        benchmark::DoNotOptimize(result.first);
        benchmark::DoNotOptimize(result.second.data());
        benchmark::DoNotOptimize(result.second.size());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * envelope.size());
}

void BM_OnsetStrengthRealAudio(benchmark::State& state) {
    const librosa::ArrayXr* y = nullptr;

    state.PauseTiming();
    try {
        y = &real_audio();
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }
    state.ResumeTiming();

    for (auto _ : state) {
        const auto start = std::chrono::steady_clock::now();
        auto envelope = librosa::onset::onset_strength(*y, kSampleRate, kNFFT, kHopLength);
        const auto end = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(end - start).count());
        benchmark::DoNotOptimize(envelope.data());
        benchmark::DoNotOptimize(envelope.size());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * y->size());
}

void BM_LoadRealAudioDefaultSampleRate(benchmark::State& state) {
    const char* path = nullptr;

    try {
        path = real_audio_path();
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }

    for (auto _ : state) {
        auto audio = librosa::load(path, kSampleRate, true);
        benchmark::DoNotOptimize(audio.samples.data());
        benchmark::DoNotOptimize(audio.samples.size());
        benchmark::DoNotOptimize(audio.sample_rate);
        benchmark::ClobberMemory();
    }
}

void BM_LoadRealAudioNativeSampleRate(benchmark::State& state) {
    const char* path = nullptr;

    try {
        path = real_audio_path();
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }

    for (auto _ : state) {
        auto audio = librosa::load(path, std::nullopt, true);
        benchmark::DoNotOptimize(audio.samples.data());
        benchmark::DoNotOptimize(audio.samples.size());
        benchmark::DoNotOptimize(audio.sample_rate);
        benchmark::ClobberMemory();
    }
}

void BM_TempoRealAudioEnvelope(benchmark::State& state) {
    librosa::ArrayXr envelope;

    state.PauseTiming();
    try {
        envelope = librosa::onset::onset_strength(real_audio(), kSampleRate, kNFFT, kHopLength);
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }
    state.ResumeTiming();

    for (auto _ : state) {
        const auto start = std::chrono::steady_clock::now();
        auto bpm = librosa::beat::tempo(envelope, kSampleRate, kHopLength);
        const auto end = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(end - start).count());
        benchmark::DoNotOptimize(bpm);
    }

    state.SetItemsProcessed(state.iterations() * envelope.size());
}

void BM_BeatTrackRealAudio(benchmark::State& state) {
    const librosa::ArrayXr* y = nullptr;

    state.PauseTiming();
    try {
        y = &real_audio();
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }
    state.ResumeTiming();

    for (auto _ : state) {
        const auto start = std::chrono::steady_clock::now();
        auto result = librosa::beat::beat_track_audio(*y, kSampleRate, kHopLength);
        const auto end = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(end - start).count());
        benchmark::DoNotOptimize(result.first);
        benchmark::DoNotOptimize(result.second.data());
        benchmark::DoNotOptimize(result.second.size());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * y->size());
}

bool register_real_audio_benchmarks() {
    if (!has_real_audio_path()) {
        return false;
    }

    benchmark::RegisterBenchmark("BM_OnsetStrengthRealAudio", &BM_OnsetStrengthRealAudio)
        ->Iterations(1)
        ->UseManualTime()
        ->Unit(benchmark::kSecond);

    benchmark::RegisterBenchmark("BM_LoadRealAudioDefaultSampleRate",
                                 &BM_LoadRealAudioDefaultSampleRate)
        ->Iterations(1)
        ->Unit(benchmark::kSecond);

    benchmark::RegisterBenchmark("BM_LoadRealAudioNativeSampleRate",
                                 &BM_LoadRealAudioNativeSampleRate)
        ->Iterations(1)
        ->Unit(benchmark::kSecond);

    benchmark::RegisterBenchmark("BM_TempoRealAudioEnvelope", &BM_TempoRealAudioEnvelope)
        ->Iterations(1)
        ->UseManualTime()
        ->Unit(benchmark::kSecond);

    benchmark::RegisterBenchmark("BM_BeatTrackRealAudio", &BM_BeatTrackRealAudio)
        ->Iterations(1)
        ->UseManualTime()
        ->Unit(benchmark::kSecond);

    return true;
}

const bool kRealAudioBenchmarksRegistered = register_real_audio_benchmarks();

} // namespace

BENCHMARK(BM_OnsetStrengthSyntheticAudio)
    ->Arg(10)
    ->Arg(30)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_TempoEnvelope)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_BeatTrackFixedTempoEnvelope)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_BeatTrackEstimatedTempoEnvelope)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Unit(benchmark::kMillisecond);
