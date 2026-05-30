#include "benchmark_helpers.hpp"

#include <librosa/decompose.hpp>
#include <librosa/effects.hpp>
#include <librosa/filters.hpp>
#include <librosa/segment.hpp>
#include <librosa/sequence.hpp>

#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace {

using namespace librosa_bench;

void BM_FiltersBanksAndWavelets(benchmark::State& state) {
    const int n_fft = static_cast<int>(state.range(0));
    const auto freqs = librosa::cqt_frequencies(48, librosa::note_to_hz("C2"));
    const auto alpha = librosa::filters::relative_bandwidth(freqs);

    for (auto _ : state) {
        auto mel = librosa::filters::mel(kSampleRate, n_fft, 64);
        auto chroma = librosa::filters::chroma(kSampleRate, n_fft, 12);
        auto bandwidth_enum = librosa::filters::window_bandwidth(librosa::WindowType::Hann);
        auto bandwidth_name = librosa::filters::window_bandwidth("hann");
        auto cq_chroma = librosa::filters::cq_to_chroma(48, 12, 12);
        auto rel = librosa::filters::relative_bandwidth(freqs);
        auto lengths = librosa::filters::wavelet_lengths(freqs, kSampleRate);
        auto lengths_alpha = librosa::filters::wavelet_lengths(
            freqs, kSampleRate, librosa::WindowType::Hann, 1.0, std::optional<librosa::Real>(0.0), alpha);
        auto wavelet = librosa::filters::wavelet(freqs, kSampleRate);
        auto wavelet_alpha = librosa::filters::wavelet(
            freqs, kSampleRate, librosa::WindowType::Hann, 1.0, true, 1.0,
            std::optional<librosa::Real>(0.0), alpha);
        auto diagonal = librosa::filters::diagonal_filter(librosa::WindowType::Hann, 21);
        auto mr = librosa::filters::mr_frequencies();

        benchmark::DoNotOptimize(bandwidth_enum);
        benchmark::DoNotOptimize(bandwidth_name);
        consume_eigen(mel);
        consume_eigen(chroma);
        consume_eigen(cq_chroma);
        consume_eigen(rel);
        consume_eigen(lengths.first);
        benchmark::DoNotOptimize(lengths.second);
        consume_eigen(lengths_alpha.first);
        benchmark::DoNotOptimize(lengths_alpha.second);
        consume_eigen(wavelet.first);
        consume_eigen(wavelet.second);
        consume_eigen(wavelet_alpha.first);
        consume_eigen(wavelet_alpha.second);
        consume_eigen(diagonal);
        consume_eigen(mr.first);
        consume_eigen(mr.second);
    }
}

void BM_FiltersSemitoneAndSOS(benchmark::State& state) {
    const auto x = make_tone(440.0, state.range(0));
    librosa::ArrayXr center_freqs(3);
    center_freqs << 220.0, 440.0, 880.0;
    librosa::ArrayXr sample_rates = librosa::ArrayXr::Constant(3, kSampleRate);
    const auto filterbank = librosa::filters::semitone_filterbank(center_freqs, 0.0, sample_rates);
    const auto& sos = filterbank.first.front();
    const auto zi = librosa::filters::sosfilt_zi(sos);

    for (auto _ : state) {
        auto filtered = librosa::filters::sosfilt(sos, x);
        auto filtered_zi = librosa::filters::sosfilt(sos, x, zi);
        auto zi_out = librosa::filters::sosfilt_zi(sos);
        auto filtfilt = librosa::filters::sosfiltfilt(sos, x);
        auto fb = librosa::filters::semitone_filterbank(center_freqs, 0.0, sample_rates);

        consume_eigen(filtered);
        consume_eigen(filtered_zi.first);
        consume_eigen(filtered_zi.second);
        consume_eigen(zi_out);
        consume_eigen(filtfilt);
        benchmark::DoNotOptimize(fb.first.size());
        consume_eigen(fb.second);
    }

    state.SetItemsProcessed(state.iterations() * x.size());
}

void BM_DecomposeHPSSAndMedian(benchmark::State& state) {
    const auto S = make_positive_matrix(64, state.range(0));
    const auto C = make_complex_matrix(64, state.range(0));

    for (auto _ : state) {
        auto median = librosa::decompose::median_filter_2d(S, {5, 5});
        auto hpss = librosa::decompose::hpss(S, 7);
        auto hpss_pair = librosa::decompose::hpss(S, std::pair<int, int>{7, 9});
        auto hpss_margin = librosa::decompose::hpss(S, 7, 2.0, false,
                                                    std::pair<librosa::Real, librosa::Real>{1.0, 2.0});
        auto complex = librosa::decompose::hpss_complex(C, 7);
        auto complex_pair = librosa::decompose::hpss_complex(C, std::pair<int, int>{7, 9});
        auto complex_margin = librosa::decompose::hpss_complex(
            C, 7, 2.0, false, std::pair<librosa::Real, librosa::Real>{1.0, 2.0});

        consume_eigen(median);
        consume_eigen(hpss.first);
        consume_eigen(hpss.second);
        consume_eigen(hpss_pair.first);
        consume_eigen(hpss_pair.second);
        consume_eigen(hpss_margin.first);
        consume_eigen(hpss_margin.second);
        consume_eigen(complex.first);
        consume_eigen(complex.second);
        consume_eigen(complex_pair.first);
        consume_eigen(complex_pair.second);
        consume_eigen(complex_margin.first);
        consume_eigen(complex_margin.second);
    }

    state.SetItemsProcessed(state.iterations() * S.size());
}

void BM_DecomposeNMFAndNNFilter(benchmark::State& state) {
    const auto S = make_positive_matrix(32, state.range(0));
    const auto rec = make_neighbor_graph(state.range(0));

    for (auto _ : state) {
        auto nmf = librosa::decompose::decompose_nmf(S, 8, false, 40, 1e-4);
        auto nn_mean = librosa::decompose::nn_filter(S, rec, librosa::AggregateFunc::Mean);
        auto nn_median = librosa::decompose::nn_filter(S, rec, librosa::AggregateFunc::Median);

        consume_eigen(nmf.first);
        consume_eigen(nmf.second);
        consume_eigen(nn_mean);
        consume_eigen(nn_median);
    }

    state.SetItemsProcessed(state.iterations() * S.size());
}

void BM_EffectsPhaseAndPitch(benchmark::State& state) {
    const auto y = make_tone(440.0, state.range(0));

    for (auto _ : state) {
        auto stretched = librosa::effects::time_stretch(y, 1.25, 1024, 256);
        auto shifted = librosa::effects::pitch_shift(y, kSampleRate, 2.0, 12,
                                                     "linear", 1024, 256);
        consume_eigen(stretched);
        consume_eigen(shifted);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_EffectsEditingAndEmphasis(benchmark::State& state) {
    auto y = make_tone(440.0, state.range(0));
    y.head(state.range(0) / 4).setZero();
    y.tail(state.range(0) / 4).setZero();
    const std::vector<std::pair<Eigen::Index, Eigen::Index>> intervals = {
        {state.range(0) / 4, state.range(0) / 2},
        {state.range(0) / 2, 3 * state.range(0) / 4},
    };

    for (auto _ : state) {
        auto trimmed = librosa::effects::trim(y, 60.0, -1.0, 1024, 256);
        auto split = librosa::effects::split(y, 60.0, -1.0, 1024, 256);
        auto pre = librosa::effects::preemphasis(y);
        auto de = librosa::effects::deemphasis(pre);
        auto remixed = librosa::effects::remix(y, intervals, true);

        consume_eigen(trimmed.first);
        benchmark::DoNotOptimize(trimmed.second.first);
        benchmark::DoNotOptimize(trimmed.second.second);
        benchmark::DoNotOptimize(split.data());
        benchmark::DoNotOptimize(split.size());
        consume_eigen(pre);
        consume_eigen(de);
        consume_eigen(remixed);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_EffectsHPSSAudio(benchmark::State& state) {
    const librosa::ArrayXr y =
        (make_tone(440.0, state.range(0)) + make_click_track(state.range(0))).eval();

    for (auto _ : state) {
        auto harmonic = librosa::effects::harmonic(y, 15, 2.0, false, 1.0, 1024, 256);
        auto percussive = librosa::effects::percussive(y, 15, 2.0, false, 1.0, 1024, 256);
        consume_eigen(harmonic);
        consume_eigen(percussive);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_SegmentSimilarity(benchmark::State& state) {
    const auto data = make_feature_matrix(12, state.range(0));
    const auto data_ref = make_feature_matrix(12, state.range(0));

    for (auto _ : state) {
        auto cross = librosa::segment::cross_similarity(data, data_ref, 8);
        auto cross_sparse = librosa::segment::cross_similarity_sparse(data, data_ref, 8);
        auto rec = librosa::segment::recurrence_matrix(data, 8, 2);
        auto rec_sparse = librosa::segment::recurrence_matrix_sparse(data, 8, 2);
        auto lag = librosa::segment::recurrence_to_lag(rec);
        auto rec2 = librosa::segment::lag_to_recurrence(lag);
        auto filtered = librosa::segment::timelag_filter(rec, [](const librosa::ArrayXXr& lag_matrix) {
            return lag_matrix;
        });

        consume_eigen(cross);
        consume_sparse(cross_sparse);
        consume_eigen(rec);
        consume_sparse(rec_sparse);
        consume_eigen(lag);
        consume_eigen(rec2);
        consume_eigen(filtered);
    }

    state.SetItemsProcessed(state.iterations() * data.size());
}

void BM_SegmentPathAndClustering(benchmark::State& state) {
    const auto data = make_feature_matrix(8, state.range(0));
    const auto rec = make_recurrence(state.range(0));
    const std::vector<Eigen::Index> frames = {0, state.range(0) / 2};

    for (auto _ : state) {
        auto enhanced = librosa::segment::path_enhance(rec, 11);
        auto bounds = librosa::segment::agglomerative(data, 4);
        auto subsegments = librosa::segment::subsegment(data, frames, 2);

        consume_eigen(enhanced);
        consume_vector(bounds);
        consume_vector(subsegments);
    }

    state.SetItemsProcessed(state.iterations() * data.size());
}

void BM_SequenceTransitionsAndViterbi(benchmark::State& state) {
    const int n_states = 8;
    const Eigen::Index n_frames = state.range(0);
    librosa::ArrayXXr prob = librosa::ArrayXXr::Constant(n_states, n_frames, 0.1);
    for (Eigen::Index t = 0; t < n_frames; ++t) {
        prob(static_cast<int>(t % n_states), t) = 0.9;
    }
    const auto trans = librosa::sequence::transition_loop(n_states, 0.85);
    librosa::ArrayXXr trans_binary(2, 2);
    trans_binary << 0.9, 0.1,
                    0.1, 0.9;
    librosa::ArrayXr p_state = librosa::ArrayXr::Constant(n_states, 1.0 / n_states);
    librosa::ArrayXr p_init = p_state;

    for (auto _ : state) {
        auto uniform = librosa::sequence::transition_uniform(n_states);
        auto loop_scalar = librosa::sequence::transition_loop(n_states, 0.85);
        auto loop_vector = librosa::sequence::transition_loop(n_states, p_state);
        auto cycle_scalar = librosa::sequence::transition_cycle(n_states, 0.85);
        auto cycle_vector = librosa::sequence::transition_cycle(n_states, p_state);
        auto local = librosa::sequence::transition_local(n_states, 5);
        auto states = librosa::sequence::viterbi(prob, trans, p_init);
        auto states_logp = librosa::sequence::viterbi_with_logp(prob, trans, p_init);
        auto discr = librosa::sequence::viterbi_discriminative(prob, trans, p_state, p_init);
        auto binary = librosa::sequence::viterbi_binary(prob, trans_binary, p_state, p_init);
        auto binary_logp = librosa::sequence::viterbi_binary_with_logp(
            prob, trans_binary, p_state, p_init);

        consume_eigen(uniform);
        consume_eigen(loop_scalar);
        consume_eigen(loop_vector);
        consume_eigen(cycle_scalar);
        consume_eigen(cycle_vector);
        consume_eigen(local);
        consume_vector(states);
        consume_vector(states_logp.first);
        benchmark::DoNotOptimize(states_logp.second);
        consume_vector(discr);
        consume_eigen(binary);
        consume_eigen(binary_logp.first);
        consume_eigen(binary_logp.second);
    }

    state.SetItemsProcessed(state.iterations() * prob.size());
}

void BM_SequenceAlignment(benchmark::State& state) {
    const auto X = make_feature_matrix(6, state.range(0));
    const auto Y = make_feature_matrix(6, state.range(0) + 16);
    const auto sim = make_recurrence(state.range(0));
    const auto steps =
        Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(state.range(0), state.range(0));

    for (auto _ : state) {
        auto euclidean = librosa::sequence::cdist_euclidean(X, Y);
        auto cosine = librosa::sequence::cdist_cosine(X, Y);
        auto dtw = librosa::sequence::dtw(X, Y);
        auto dtw_backtrack = librosa::sequence::dtw_backtrack(X, Y);
        auto path = librosa::sequence::dtw_backtracking(steps);
        auto rqa = librosa::sequence::rqa(sim);
        auto rqa_backtrack = librosa::sequence::rqa_backtrack(sim);

        consume_eigen(euclidean);
        consume_eigen(cosine);
        consume_eigen(dtw);
        consume_eigen(dtw_backtrack.first);
        consume_vector(dtw_backtrack.second);
        consume_vector(path);
        consume_eigen(rqa);
        consume_eigen(rqa_backtrack.first);
        consume_vector(rqa_backtrack.second);
    }

    state.SetItemsProcessed(state.iterations() * X.size());
}

} // namespace

BENCHMARK(BM_FiltersBanksAndWavelets)->Arg(1024)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FiltersSemitoneAndSOS)->Arg(2048)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_DecomposeHPSSAndMedian)->Arg(96)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_DecomposeNMFAndNNFilter)->Arg(64)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_EffectsPhaseAndPitch)->Arg(4096)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_EffectsEditingAndEmphasis)->Arg(8192)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_EffectsHPSSAudio)->Arg(4096)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SegmentSimilarity)->Arg(96)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SegmentPathAndClustering)->Arg(96)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SequenceTransitionsAndViterbi)->Arg(256)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SequenceAlignment)->Arg(96)->Unit(benchmark::kMillisecond);
