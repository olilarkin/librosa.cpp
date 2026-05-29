#include "benchmark_helpers.hpp"

#include <librosa/core/constantq.hpp>
#include <librosa/core/convert.hpp>
#include <librosa/core/harmonic.hpp>
#include <librosa/core/intervals.hpp>
#include <librosa/core/notation.hpp>
#include <librosa/core/pitch.hpp>
#include <librosa/core/spectrum.hpp>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace librosa_bench;

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

void BM_AudioToMono(benchmark::State& state) {
    librosa::ArrayXXr stereo(2, state.range(0));
    stereo.row(0) = make_tone(440.0, state.range(0)).transpose();
    stereo.row(1) = make_tone(660.0, state.range(0)).transpose();

    for (auto _ : state) {
        auto mono = librosa::to_mono(stereo);
        consume_eigen(mono);
    }

    state.SetItemsProcessed(state.iterations() * stereo.cols());
}

void BM_AudioResampleLinear(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));

    for (auto _ : state) {
        auto resampled = librosa::resample(y, kSampleRate, 11025.0, "linear");
        consume_eigen(resampled);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_AudioResampleFFT(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));

    for (auto _ : state) {
        auto resampled = librosa::resample(y, kSampleRate, 11025.0, "fft");
        consume_eigen(resampled);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_AudioResampleKaiserHQ(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));

    for (auto _ : state) {
        auto resampled = librosa::resample(y, kSampleRate, 11025.0, "kaiser_hq");
        consume_eigen(resampled);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_AudioAutocorrelate(benchmark::State& state) {
    const auto y = make_random_vector(state.range(0));
    const auto matrix = make_feature_matrix(8, state.range(0) / 8);

    for (auto _ : state) {
        auto ac = librosa::autocorrelate(y, 512);
        auto ac2 = librosa::autocorrelate(matrix, 32, -1);
        consume_eigen(ac);
        consume_eigen(ac2);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_AudioLPC(benchmark::State& state) {
    const auto y = make_tone(220.0, state.range(0));
    const auto matrix = make_feature_matrix(4, state.range(0) / 4);

    for (auto _ : state) {
        auto coeffs = librosa::lpc(y, 16);
        auto coeffs2 = librosa::lpc(matrix, 8, -1);
        consume_eigen(coeffs);
        consume_eigen(coeffs2);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_AudioZeroCrossings(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));
    const auto matrix = make_feature_matrix(4, state.range(0) / 4);

    for (auto _ : state) {
        auto crossings = librosa::zero_crossings(y);
        auto crossings2 = librosa::zero_crossings(matrix, 1e-10, std::nullopt, true, true, -1);
        consume_eigen(crossings);
        consume_eigen(crossings2);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_AudioSynthesisAndMuLaw(benchmark::State& state) {
    const Eigen::Index n = state.range(0);
    librosa::ArrayXr times(8);
    times.setLinSpaced(0.0, 1.0);
    std::vector<Eigen::Index> frames = {0, 8, 16, 24, 32, 40, 48, 56};

    for (auto _ : state) {
        auto clicks = librosa::clicks(times, kSampleRate, 1000.0, 0.05, static_cast<int>(n));
        auto frame_clicks = librosa::clicks_frames(frames, kSampleRate, kHopLength,
                                                   1000.0, 0.05, static_cast<int>(n));
        auto tone = librosa::tone(440.0, kSampleRate, static_cast<int>(n));
        auto chirp = librosa::chirp(110.0, 1760.0, kSampleRate, static_cast<int>(n));
        auto compressed = librosa::mu_compress(tone);
        auto expanded = librosa::mu_expand(compressed);
        consume_eigen(clicks);
        consume_eigen(frame_clicks);
        consume_eigen(tone);
        consume_eigen(chirp);
        consume_eigen(compressed);
        consume_eigen(expanded);
    }

    state.SetItemsProcessed(state.iterations() * n);
}

void BM_AudioDuration(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));
    const auto D = librosa::stft(y, 1024, 256);
    const auto S = librosa::magnitude(D);

    for (auto _ : state) {
        auto from_audio = librosa::get_duration(y, kSampleRate);
        auto from_spectrogram = librosa::get_duration(S, kSampleRate, 256, 1024, true);
        benchmark::DoNotOptimize(from_audio);
        benchmark::DoNotOptimize(from_spectrogram);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_AudioFileMetadataRealAudio(benchmark::State& state) {
    const char* path = nullptr;

    try {
        path = real_audio_path();
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        return;
    }

    for (auto _ : state) {
        auto duration = librosa::get_duration(path);
        auto sample_rate = librosa::get_samplerate(path);
        auto info = librosa::get_audio_info(path);
        benchmark::DoNotOptimize(duration);
        benchmark::DoNotOptimize(sample_rate);
        benchmark::DoNotOptimize(info.samples);
        benchmark::DoNotOptimize(info.sample_rate);
        benchmark::DoNotOptimize(info.channels);
        benchmark::DoNotOptimize(info.duration);
    }
}

void BM_ConvertFrameSampleTime(benchmark::State& state) {
    const auto values = librosa::ArrayXr::LinSpaced(state.range(0), 0.0,
                                                    static_cast<librosa::Real>(state.range(0) - 1));
    const auto matrix = make_feature_matrix(8, state.range(0));

    for (auto _ : state) {
        auto samples = librosa::frames_to_samples(values, kHopLength, kNFFT);
        auto frames = librosa::samples_to_frames(samples, kHopLength, kNFFT);
        auto times = librosa::frames_to_time(frames, kSampleRate, kHopLength, kNFFT);
        auto frames2 = librosa::time_to_frames(times, kSampleRate, kHopLength, kNFFT);
        auto samples2 = librosa::time_to_samples(times, kSampleRate);
        auto times2 = librosa::samples_to_time(samples2, kSampleRate);
        auto block_frames = librosa::blocks_to_frames(values, 4);
        auto block_samples = librosa::blocks_to_samples(values, 4, kHopLength);
        auto block_times = librosa::blocks_to_time(values, 4, kHopLength, kSampleRate);
        auto samples_like = librosa::samples_like(matrix, kHopLength, std::nullopt, -1);
        auto times_like = librosa::times_like(matrix, kSampleRate, kHopLength, std::nullopt, -1);

        benchmark::DoNotOptimize(librosa::frames_to_samples(Eigen::Index{128}, kHopLength, kNFFT));
        benchmark::DoNotOptimize(librosa::samples_to_frames(Eigen::Index{65536}, kHopLength, kNFFT));
        benchmark::DoNotOptimize(librosa::frames_to_time(Eigen::Index{128}, kSampleRate, kHopLength, kNFFT));
        benchmark::DoNotOptimize(librosa::time_to_frames(1.25, kSampleRate, kHopLength, kNFFT));
        benchmark::DoNotOptimize(librosa::time_to_samples(1.25, kSampleRate));
        benchmark::DoNotOptimize(librosa::samples_to_time(Eigen::Index{65536}, kSampleRate));
        benchmark::DoNotOptimize(librosa::blocks_to_frames(Eigen::Index{16}, 4));
        benchmark::DoNotOptimize(librosa::blocks_to_samples(Eigen::Index{16}, 4, kHopLength));
        benchmark::DoNotOptimize(librosa::blocks_to_time(Eigen::Index{16}, 4, kHopLength, kSampleRate));

        consume_eigen(samples);
        consume_eigen(frames);
        consume_eigen(times);
        consume_eigen(frames2);
        consume_eigen(samples2);
        consume_eigen(times2);
        consume_eigen(block_frames);
        consume_eigen(block_samples);
        consume_eigen(block_times);
        consume_eigen(samples_like);
        consume_eigen(times_like);
    }

    state.SetItemsProcessed(state.iterations() * values.size());
}

void BM_ConvertPitchScales(benchmark::State& state) {
    const auto midi = librosa::ArrayXr::LinSpaced(state.range(0), 24.0, 96.0);
    const auto freqs = librosa::ArrayXr::LinSpaced(state.range(0), 55.0, 1760.0);
    std::vector<std::string> notes = {"C2", "E2", "G2", "B2", "C3", "E3",
                                      "G3", "B3", "C4", "E4", "G4", "B4"};

    for (auto _ : state) {
        auto hz = librosa::midi_to_hz(midi);
        auto midi_back = librosa::hz_to_midi(freqs);
        auto note_midi = librosa::note_to_midi(notes);
        auto note_hz = librosa::note_to_hz(notes);
        auto midi_notes = librosa::midi_to_note(midi, true, false, false);
        auto hz_notes = librosa::hz_to_note(freqs, true, false, false);
        auto mel = librosa::hz_to_mel(freqs);
        auto hz_back = librosa::mel_to_hz(mel);
        auto octs = librosa::hz_to_octs(freqs);
        auto hz_from_octs = librosa::octs_to_hz(octs);
        auto tuning = librosa::A4_to_tuning(freqs);
        auto a4 = librosa::tuning_to_A4(tuning);

        benchmark::DoNotOptimize(librosa::midi_to_hz(69.0));
        benchmark::DoNotOptimize(librosa::hz_to_midi(440.0));
        benchmark::DoNotOptimize(librosa::note_to_midi("A4"));
        benchmark::DoNotOptimize(librosa::note_to_hz("A4"));
        benchmark::DoNotOptimize(librosa::midi_to_note(69.0, true, false, false));
        benchmark::DoNotOptimize(librosa::hz_to_note(440.0, true, false, false));
        benchmark::DoNotOptimize(librosa::hz_to_mel(1000.0));
        benchmark::DoNotOptimize(librosa::mel_to_hz(100.0));
        benchmark::DoNotOptimize(librosa::hz_to_octs(440.0));
        benchmark::DoNotOptimize(librosa::octs_to_hz(4.0));
        benchmark::DoNotOptimize(librosa::A4_to_tuning(440.0));
        benchmark::DoNotOptimize(librosa::tuning_to_A4(0.0));

        consume_eigen(hz);
        consume_eigen(midi_back);
        consume_eigen(note_midi);
        consume_eigen(note_hz);
        consume_vector(midi_notes);
        consume_vector(hz_notes);
        consume_eigen(mel);
        consume_eigen(hz_back);
        consume_eigen(octs);
        consume_eigen(hz_from_octs);
        consume_eigen(tuning);
        consume_eigen(a4);
    }

    state.SetItemsProcessed(state.iterations() * midi.size());
}

void BM_ConvertFrequencyGridsAndWeighting(benchmark::State& state) {
    const Eigen::Index n = state.range(0);
    const auto freqs = librosa::ArrayXr::LinSpaced(n, 20.0, 20000.0);

    for (auto _ : state) {
        auto fft_freqs = librosa::fft_frequencies(kSampleRate, kNFFT);
        auto cqt_freqs = librosa::cqt_frequencies(static_cast<int>(n), librosa::note_to_hz("C1"));
        auto mel_freqs = librosa::mel_frequencies(static_cast<int>(n), 0.0, kSampleRate / 2.0);
        auto tempo_freqs = librosa::tempo_frequencies(static_cast<int>(n), kHopLength, kSampleRate);
        auto fourier_tempo = librosa::fourier_tempo_frequencies(kSampleRate, 384, kHopLength);
        auto a = librosa::A_weighting(freqs);
        auto b = librosa::B_weighting(freqs);
        auto c = librosa::C_weighting(freqs);
        auto d = librosa::D_weighting(freqs);
        auto z = librosa::Z_weighting(freqs);
        auto combined = librosa::frequency_weighting(freqs, librosa::WeightType::A);

        benchmark::DoNotOptimize(librosa::A_weighting(1000.0));
        benchmark::DoNotOptimize(librosa::B_weighting(1000.0));
        benchmark::DoNotOptimize(librosa::C_weighting(1000.0));
        benchmark::DoNotOptimize(librosa::D_weighting(1000.0));
        benchmark::DoNotOptimize(librosa::Z_weighting(1000.0));

        consume_eigen(fft_freqs);
        consume_eigen(cqt_freqs);
        consume_eigen(mel_freqs);
        consume_eigen(tempo_freqs);
        consume_eigen(fourier_tempo);
        consume_eigen(a);
        consume_eigen(b);
        consume_eigen(c);
        consume_eigen(d);
        consume_eigen(z);
        consume_eigen(combined);
    }

    state.SetItemsProcessed(state.iterations() * n);
}

void BM_ConvertNotation(benchmark::State& state) {
    for (auto _ : state) {
        auto thaat = librosa::thaat_to_degrees("bilaval");
        auto mela = librosa::mela_to_degrees(29);
        auto mela_name = librosa::mela_to_degrees("dheerasankarabharanam");
        auto svara = librosa::mela_to_svara(29, true, false);
        auto svara_name = librosa::mela_to_svara("dheerasankarabharanam", true, false);
        auto mela_list = librosa::list_mela();
        auto thaat_list = librosa::list_thaat();
        auto key_degrees = librosa::key_to_degrees("C:maj");
        auto key_notes = librosa::key_to_notes("C:maj", false);
        auto fifth = librosa::fifths_to_note("C", 1, false);
        auto fjs = librosa::interval_to_fjs(3.0 / 2.0, "C", 65.0 / 63.0, false);
        auto svara_h = librosa::midi_to_svara_h(67.0, 60.0, true, true, false);
        auto svara_c = librosa::midi_to_svara_c(67.0, 60.0, 29, true, false);
        auto svara_c_name = librosa::midi_to_svara_c(67.0, 60.0, "dheerasankarabharanam", true, false);
        auto hz_svara_h = librosa::hz_to_svara_h(440.0, 261.63, true, true, false);
        auto hz_svara_c = librosa::hz_to_svara_c(440.0, 261.63, 29, true, false);
        auto hz_svara_c_name = librosa::hz_to_svara_c(440.0, 261.63, "dheerasankarabharanam", true, false);
        auto note_svara_h = librosa::note_to_svara_h("G4", "C4", true, true, false);
        auto note_svara_c = librosa::note_to_svara_c("G4", "C4", 29, true, false);
        auto note_svara_c_name = librosa::note_to_svara_c("G4", "C4", "dheerasankarabharanam", true, false);
        auto hz_fjs = librosa::hz_to_fjs(660.0, 440.0, "A", false);

        consume_vector(thaat);
        consume_vector(mela);
        consume_vector(mela_name);
        consume_vector(svara);
        consume_vector(svara_name);
        benchmark::DoNotOptimize(mela_list.size());
        consume_vector(thaat_list);
        consume_vector(key_degrees);
        consume_vector(key_notes);
        benchmark::DoNotOptimize(fifth);
        benchmark::DoNotOptimize(fjs);
        benchmark::DoNotOptimize(svara_h);
        benchmark::DoNotOptimize(svara_c);
        benchmark::DoNotOptimize(svara_c_name);
        benchmark::DoNotOptimize(hz_svara_h);
        benchmark::DoNotOptimize(hz_svara_c);
        benchmark::DoNotOptimize(hz_svara_c_name);
        benchmark::DoNotOptimize(note_svara_h);
        benchmark::DoNotOptimize(note_svara_c);
        benchmark::DoNotOptimize(note_svara_c_name);
        benchmark::DoNotOptimize(hz_fjs);
    }
}

void BM_Intervals(benchmark::State& state) {
    for (auto _ : state) {
        auto pythagorean = librosa::pythagorean_intervals(12);
        auto plimit = librosa::plimit_intervals({3, 5, 7}, 12);
        auto equal = librosa::interval_frequencies(48, 55.0, "equal", 12);
        auto pyth = librosa::interval_frequencies(48, 55.0, "pythagorean", 12);
        auto ji = librosa::interval_frequencies(48, 55.0, "ji5", 12);
        auto custom = librosa::interval_frequencies(48, 55.0, pythagorean);

        consume_eigen(pythagorean);
        consume_eigen(plimit);
        consume_eigen(equal);
        consume_eigen(pyth);
        consume_eigen(ji);
        consume_eigen(custom);
    }
}

void BM_SpectrumSTFTISTFT(benchmark::State& state) {
    const auto y = make_random_vector(state.range(0));
    const auto window = librosa::get_window(librosa::WindowType::Hann, kNFFT);

    for (auto _ : state) {
        auto D = librosa::stft(y, kNFFT, kHopLength);
        auto D_window = librosa::stft(y, kNFFT, kHopLength, window);
        auto reconstructed = librosa::istft(D, kHopLength, std::nullopt, kNFFT,
                                             librosa::WindowType::Hann, true,
                                             static_cast<int>(y.size()));
        consume_eigen(D);
        consume_eigen(D_window);
        consume_eigen(reconstructed);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_SpectrumMagnitudePhaseAndDb(benchmark::State& state) {
    const auto y = make_random_vector(state.range(0));
    const auto D = librosa::stft(y, kNFFT, kHopLength);
    const librosa::ArrayXXr S = (librosa::magnitude(D).square() + 1e-6).eval();
    const auto freqs = librosa::fft_frequencies(kSampleRate, kNFFT);

    for (auto _ : state) {
        auto magphase = librosa::magphase(D);
        auto mag = librosa::magnitude(D);
        auto ph = librosa::phase(D);
        auto power_db = librosa::power_to_db(S);
        auto power_db_fn = librosa::power_to_db(S, [](const librosa::ArrayXXr& x) {
            return x.maxCoeff();
        });
        auto power = librosa::db_to_power(power_db);
        auto amp_db = librosa::amplitude_to_db(mag);
        auto amp_db_fn = librosa::amplitude_to_db(mag, [](const librosa::ArrayXXr& x) {
            return x.maxCoeff();
        });
        auto amp = librosa::db_to_amplitude(amp_db);
        auto weighted = librosa::perceptual_weighting(S, freqs);

        benchmark::DoNotOptimize(librosa::power_to_db(1.0));
        benchmark::DoNotOptimize(librosa::db_to_power(0.0));
        benchmark::DoNotOptimize(librosa::amplitude_to_db(1.0));
        benchmark::DoNotOptimize(librosa::db_to_amplitude(0.0));

        consume_eigen(magphase.first);
        consume_eigen(magphase.second);
        consume_eigen(mag);
        consume_eigen(ph);
        consume_eigen(power_db);
        consume_eigen(power_db_fn);
        consume_eigen(power);
        consume_eigen(amp_db);
        consume_eigen(amp_db_fn);
        consume_eigen(amp);
        consume_eigen(weighted);
    }

    state.SetItemsProcessed(state.iterations() * D.size());
}

void BM_SpectrumPhaseVocoderAndGriffinLim(benchmark::State& state) {
    const auto y = make_random_vector(state.range(0));
    const auto D = librosa::stft(y, 1024, 256);
    const auto S = librosa::magnitude(D);

    for (auto _ : state) {
        auto stretched = librosa::phase_vocoder(D, 1.5, 256, 1024);
        auto reconstructed = librosa::griffinlim(S, 4, 256, std::nullopt, 1024,
                                                 librosa::WindowType::Hann, true,
                                                 static_cast<int>(y.size()),
                                                 librosa::PadMode::Constant, 0.99,
                                                 "random", 123);
        consume_eigen(stretched);
        consume_eigen(reconstructed);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_SpectrumPCENWindowIIRTReassignedFMT(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));
    const auto D = librosa::stft(y, 1024, 256);
    const librosa::ArrayXXr S = (librosa::magnitude(D).square() + 1e-6).eval();
    const auto window = librosa::get_window("hann", 1024);

    for (auto _ : state) {
        auto pcen = librosa::pcen(S, kSampleRate, 256);
        auto window_enum = librosa::get_window(librosa::WindowType::Hann, 1024);
        auto window_string = librosa::get_window("hann", 1024);
        auto sumsquare = librosa::window_sumsquare(window, static_cast<int>(S.cols()), 256, 1024);
        auto iirt = librosa::iirt(y, kSampleRate, 1024, 256);
        auto reassigned = librosa::reassigned_spectrogram(y, kSampleRate, 1024, 256, std::nullopt);
        auto fmt = librosa::fmt(y, 0.5, 128, "linear");

        consume_eigen(pcen);
        consume_eigen(window_enum);
        consume_eigen(window_string);
        consume_eigen(sumsquare);
        consume_eigen(iirt);
        consume_eigen(std::get<0>(reassigned));
        consume_eigen(std::get<1>(reassigned));
        consume_eigen(std::get<2>(reassigned));
        consume_eigen(fmt);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_ConstantQForward(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));

    for (auto _ : state) {
        auto C = librosa::cqt(y, kSampleRate, kHopLength, std::nullopt, 24, 12, 0.0);
        auto V = librosa::vqt(y, kSampleRate, kHopLength, std::nullopt, 24, std::nullopt, 12, 0.0);
        auto P = librosa::pseudo_cqt(y, kSampleRate, kHopLength, std::nullopt, 24, 12, 0.0);
        auto H = librosa::hybrid_cqt(y, kSampleRate, kHopLength, std::nullopt, 24, 12, 0.0);
        consume_eigen(C);
        consume_eigen(V);
        consume_eigen(P);
        consume_eigen(H);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_ConstantQInverse(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));
    const auto C = librosa::cqt(y, kSampleRate, kHopLength, std::nullopt, 24, 12, 0.0);
    const auto magnitude = C.abs();

    for (auto _ : state) {
        auto inverted = librosa::icqt(C, kSampleRate, kHopLength, std::nullopt, 12, 0.0,
                                      1.0, 1.0, 0.01, librosa::WindowType::Hann,
                                      true, static_cast<int>(y.size()));
        auto griffinlim = librosa::griffinlim_cqt(magnitude, 2, kSampleRate, kHopLength,
                                                  std::nullopt, 12, 0.0, 1.0, 1.0,
                                                  0.01, librosa::WindowType::Hann,
                                                  true, librosa::PadMode::Constant,
                                                  static_cast<int>(y.size()), 0.99,
                                                  "random", 123);
        consume_eigen(inverted);
        consume_eigen(griffinlim);
    }

    state.SetItemsProcessed(state.iterations() * C.size());
}

void BM_PitchTuningAndTracking(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));
    const auto D = librosa::stft(y, 1024, 256);
    const auto S = librosa::magnitude(D);
    const auto freqs = librosa::cqt_frequencies(36, librosa::note_to_hz("C2"));

    for (auto _ : state) {
        auto tuning = librosa::pitch_tuning(freqs);
        auto estimate_audio = librosa::estimate_tuning(y, kSampleRate, 1024, 0.05, 12, 100.0, 1000.0);
        auto estimate_spec = librosa::estimate_tuning(S, kSampleRate, 1024, 0.05, 12, 100.0, 1000.0);
        auto pip_audio = librosa::piptrack(y, kSampleRate, 1024, 256, 100.0, 1000.0);
        auto pip_spec = librosa::piptrack(S, kSampleRate, 1024, 256, 100.0, 1000.0);
        auto yin = librosa::yin(y, 100.0, 1000.0, kSampleRate, 1024, 256);
        auto pyin = librosa::pyin(y, 100.0, 1000.0, kSampleRate, 1024, 256,
                                  32, 2.0, 18.0, 2.0, 0.1);

        benchmark::DoNotOptimize(tuning);
        benchmark::DoNotOptimize(estimate_audio);
        benchmark::DoNotOptimize(estimate_spec);
        consume_eigen(pip_audio.first);
        consume_eigen(pip_audio.second);
        consume_eigen(pip_spec.first);
        consume_eigen(pip_spec.second);
        consume_eigen(yin);
        consume_eigen(pyin.f0);
        consume_eigen(pyin.voiced_flag);
        consume_eigen(pyin.voiced_prob);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_HarmonicCore(benchmark::State& state) {
    const auto S = make_positive_matrix(state.range(0), 64);
    const auto freqs = librosa::ArrayXr::LinSpaced(state.range(0), 55.0, 4000.0);
    const auto spectrum = S.col(0).eval();
    const auto f0 = librosa::ArrayXr::Constant(S.cols(), 220.0);
    const std::vector<librosa::Real> harmonics = {1.0, 2.0, 3.0, 4.0};
    const std::vector<librosa::Real> weights = {1.0, 0.5, 0.25, 0.125};

    for (auto _ : state) {
        auto interp2d = librosa::core::interp_harmonics(S, freqs, harmonics);
        auto interp1d = librosa::core::interp_harmonics(spectrum, freqs, harmonics);
        auto f0_harmonics = librosa::core::f0_harmonics(S, f0, freqs, harmonics);
        auto salience = librosa::core::salience(S, freqs, harmonics, weights);
        consume_eigen(interp2d);
        consume_eigen(interp1d);
        consume_eigen(f0_harmonics);
        consume_eigen(salience);
    }

    state.SetItemsProcessed(state.iterations() * S.size());
}

bool register_real_audio_benchmarks() {
    if (!has_real_audio_path()) {
        return false;
    }

    benchmark::RegisterBenchmark("BM_AudioFileMetadataRealAudio",
                                 &BM_AudioFileMetadataRealAudio)
        ->Iterations(1)
        ->Unit(benchmark::kSecond);

    return true;
}

const bool kRealAudioBenchmarksRegistered = register_real_audio_benchmarks();

} // namespace

BENCHMARK(BM_AudioToMono)->Arg(22050)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AudioResampleLinear)->Arg(22050)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AudioResampleFFT)->Arg(8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AudioResampleKaiserHQ)->Arg(4096)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AudioAutocorrelate)->Arg(4096)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AudioLPC)->Arg(4096)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AudioZeroCrossings)->Arg(22050)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AudioSynthesisAndMuLaw)->Arg(22050)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AudioDuration)->Arg(22050)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_ConvertFrameSampleTime)->Arg(4096)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ConvertPitchScales)->Arg(4096)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ConvertFrequencyGridsAndWeighting)->Arg(256)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ConvertNotation)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Intervals)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SpectrumSTFTISTFT)->Arg(22050)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SpectrumMagnitudePhaseAndDb)->Arg(8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SpectrumPhaseVocoderAndGriffinLim)->Arg(8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SpectrumPCENWindowIIRTReassignedFMT)->Arg(8192)->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ConstantQForward)->Arg(8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ConstantQInverse)->Arg(8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_PitchTuningAndTracking)->Arg(8192)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HarmonicCore)->Arg(128)->Unit(benchmark::kMicrosecond);
