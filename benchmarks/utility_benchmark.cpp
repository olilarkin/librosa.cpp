#include "benchmark_helpers.hpp"

#include <librosa/util/utils.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace {

using namespace librosa_bench;

void BM_UtilValidationAndShape(benchmark::State& state) {
    const auto y = make_random_vector(state.range(0));
    const auto matrix = make_feature_matrix(16, state.range(0));
    librosa::ArrayXXr intervals(3, 2);
    intervals << 0.0, 1.0,
                 1.0, 2.0,
                 2.0, 4.0;
    std::vector<Eigen::Index> frames = {4, 12, 18, 32, 48};

    for (auto _ : state) {
        auto valid_y = librosa::util::valid_audio(y);
        auto valid_matrix = librosa::util::valid_audio(matrix);
        auto positive = librosa::util::is_positive_int(4);
        auto integer = librosa::util::valid_int(12.75);
        auto valid_intervals = librosa::util::valid_intervals(intervals);
        auto tiny = librosa::util::tiny<librosa::Real>();
        auto padded = librosa::util::pad_center(y, y.size() + 128);
        auto padded_matrix = librosa::util::pad_center(matrix, matrix.cols() + 16, -1);
        auto fixed = librosa::util::fix_length(y, y.size() + 128);
        auto fixed_matrix = librosa::util::fix_length(matrix, matrix.cols() + 16, -1);
        auto fixed_frames = librosa::util::fix_frames(frames, 0, state.range(0), true);
        auto framed = librosa::util::frame(y, 256, 64);

        benchmark::DoNotOptimize(valid_y);
        benchmark::DoNotOptimize(valid_matrix);
        benchmark::DoNotOptimize(positive);
        benchmark::DoNotOptimize(integer);
        benchmark::DoNotOptimize(valid_intervals);
        benchmark::DoNotOptimize(tiny);
        consume_eigen(padded);
        consume_eigen(padded_matrix);
        consume_eigen(fixed);
        consume_eigen(fixed_matrix);
        consume_vector(fixed_frames);
        consume_eigen(framed);
    }

    state.SetItemsProcessed(state.iterations() * y.size());
}

void BM_UtilNormalizePeaksAndSort(benchmark::State& state) {
    const auto y = make_onset_envelope(state.range(0));
    const auto matrix = make_positive_matrix(32, state.range(0));

    for (auto _ : state) {
        auto norm_vector = librosa::util::normalize(y);
        auto norm_matrix = librosa::util::normalize(matrix, 2.0, -1);
        auto max_vector = librosa::util::localmax(y);
        auto max_matrix = librosa::util::localmax(matrix, -1);
        auto min_vector = librosa::util::localmin(y);
        auto min_matrix = librosa::util::localmin(matrix, -1);
        auto peaks = librosa::util::peak_pick(y, 3, 3, 3, 3, 0.1, 3);
        auto sorted = librosa::util::axis_sort(matrix, -1);
        auto sorted_index = librosa::util::axis_sort_with_index(matrix, -1);
        auto sparse = librosa::util::sparsify_rows(matrix, 0.1);

        consume_eigen(norm_vector);
        consume_eigen(norm_matrix);
        consume_eigen(max_vector);
        consume_eigen(max_matrix);
        consume_eigen(min_vector);
        consume_eigen(min_matrix);
        consume_vector(peaks);
        consume_eigen(sorted);
        consume_eigen(sorted_index.first);
        consume_vector(sorted_index.second);
        consume_eigen(sparse);
    }

    state.SetItemsProcessed(state.iterations() * matrix.size());
}

void BM_UtilMaskSyncComplexAndStack(benchmark::State& state) {
    const auto x = make_positive_matrix(16, state.range(0));
    const librosa::ArrayXXr x_ref = (make_positive_matrix(16, state.range(0)) + 0.5).eval();
    const auto complex = make_complex_matrix(16, state.range(0));
    const auto angles = make_random_vector(state.range(0));
    const auto mags = librosa::ArrayXr::Ones(state.range(0));
    std::vector<Eigen::Index> idx = {0, state.range(0) / 4, state.range(0) / 2,
                                     3 * state.range(0) / 4};
    std::vector<librosa::ArrayXr> arrays = {
        make_random_vector(state.range(0)),
        make_random_vector(state.range(0)),
        make_random_vector(state.range(0)),
    };

    for (auto _ : state) {
        auto mask = librosa::util::softmask(x, x_ref, 2.0);
        auto sync = librosa::util::sync(x, idx, librosa::AggregateFunc::Mean, true, -1);
        auto abs2_matrix = librosa::util::abs2(complex);
        auto abs2_vector = librosa::util::abs2(complex.col(0).eval());
        auto phasor_mag = librosa::util::phasor(angles, mags);
        auto phasor_scalar = librosa::util::phasor(angles, 1.0);
        auto gradient = librosa::util::cyclic_gradient(angles);
        auto filled = librosa::util::fill_off_diagonal(make_recurrence(64), 0.0);
        auto stacked_rows = librosa::util::stack(arrays, 0);
        auto stacked_cols = librosa::util::stack(arrays, 1);

        consume_eigen(mask);
        consume_eigen(sync);
        consume_eigen(abs2_matrix);
        consume_eigen(abs2_vector);
        consume_eigen(phasor_mag);
        consume_eigen(phasor_scalar);
        consume_eigen(gradient);
        consume_eigen(filled);
        consume_eigen(stacked_rows);
        consume_eigen(stacked_cols);
    }

    state.SetItemsProcessed(state.iterations() * x.size());
}

void BM_UtilBufferUniquenessShearIntervalsNNLS(benchmark::State& state) {
    const auto matrix = make_positive_matrix(32, state.range(0));
    const auto A_array = make_positive_matrix(48, 16);
    const librosa::MatrixXr A = A_array.matrix();
    const auto B = make_positive_matrix(48, 8);
    std::vector<int16_t> pcm(static_cast<size_t>(state.range(0)));
    for (Eigen::Index i = 0; i < state.range(0); ++i) {
        pcm[static_cast<size_t>(i)] = static_cast<int16_t>((i % 1024) - 512);
    }
    const auto unique = librosa::ArrayXr::LinSpaced(state.range(0), 0.0,
                                                    static_cast<librosa::Real>(state.range(0) - 1));
    librosa::ArrayXXr intervals_from(3, 2);
    intervals_from << 0.0, 1.0,
                      1.0, 2.0,
                      2.0, 3.0;
    librosa::ArrayXXr intervals_to(3, 2);
    intervals_to << 0.0, 1.5,
                    1.5, 2.5,
                    2.5, 4.0;

    for (auto _ : state) {
        auto floats = librosa::util::buf_to_float(pcm.data(), pcm.size(), 2);
        auto is_unique = librosa::util::is_unique(unique);
        auto unique_count = librosa::util::count_unique(unique);
        auto sheared_time = librosa::util::shear(matrix, 1, -1);
        auto sheared_freq = librosa::util::shear(matrix, 1, 0);
        auto slice = librosa::util::index_to_slice(8, 0, 64, 2, true);
        auto matches = librosa::util::match_intervals(intervals_from, intervals_to, true);
        auto solution = librosa::util::nnls(A, B);

        consume_eigen(floats);
        benchmark::DoNotOptimize(is_unique);
        benchmark::DoNotOptimize(unique_count);
        consume_eigen(sheared_time);
        consume_eigen(sheared_freq);
        consume_vector(slice);
        consume_vector(matches);
        consume_eigen(solution);
    }

    state.SetItemsProcessed(state.iterations() * matrix.size());
}

} // namespace

BENCHMARK(BM_UtilValidationAndShape)->Arg(2048)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_UtilNormalizePeaksAndSort)->Arg(256)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_UtilMaskSyncComplexAndStack)->Arg(256)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_UtilBufferUniquenessShearIntervalsNNLS)->Arg(128)->Unit(benchmark::kMicrosecond);
