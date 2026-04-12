#include <gtest/gtest.h>
#include <librosa/util/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>
#include <random>

using namespace librosa;
using namespace librosa::util;

namespace {

// Random seed utility
std::mt19937 rng(628318530);

void srand_reset() {
    rng.seed(628318530);
}

ArrayXr random_array(Eigen::Index size) {
    std::normal_distribution<Real> dist(0.0, 1.0);
    ArrayXr result(size);
    for (Eigen::Index i = 0; i < size; ++i) {
        result(i) = dist(rng);
    }
    return result;
}

} // anonymous namespace

// ============================================================================
// Frame Tests
// ============================================================================

class FrameTest : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(FrameTest, Frame1D) {
    auto [frame_length, hop_length] = GetParam();
    ArrayXr y = random_array(32);

    ArrayXXr y_frame = frame(y, frame_length, hop_length);

    // Check dimensions: (frame_length, n_frames)
    EXPECT_EQ(y_frame.rows(), frame_length);

    // Verify each frame matches the original signal
    for (Eigen::Index i = 0; i < y_frame.cols(); ++i) {
        for (int j = 0; j < frame_length; ++j) {
            EXPECT_NEAR(y_frame(j, i), y(i * hop_length + j), 1e-10);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(FrameTests, FrameTest,
    ::testing::Values(
        std::make_tuple(4, 2),
        std::make_tuple(4, 4),
        std::make_tuple(8, 2),
        std::make_tuple(8, 4)
    ));

TEST(FrameTest, FrameTooShortThrows) {
    ArrayXr x = ArrayXr::LinSpaced(16, 0, 15);
    EXPECT_THROW(frame(x, 17, 1), ParameterError);
}

TEST(FrameTest, BadHopLengthThrows) {
    ArrayXr x = ArrayXr::LinSpaced(16, 0, 15);
    EXPECT_THROW(frame(x, 4, 0), ParameterError);
}

// ============================================================================
// Pad Center Tests
// ============================================================================

class PadCenterTest : public ::testing::TestWithParam<std::tuple<int, PadMode>> {};

TEST_P(PadCenterTest, PadCenter1D) {
    auto [extra, mode] = GetParam();
    ArrayXr y = ArrayXr::Ones(16);
    Eigen::Index n = extra + y.size();

    ArrayXr y_out = pad_center(y, n, mode);

    Eigen::Index n_pad = (n - y.size()) / 2;

    // Check padded segment matches original
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y_out(n_pad + i), y(i), 1e-10);
    }
}

INSTANTIATE_TEST_SUITE_P(PadCenterTests, PadCenterTest,
    ::testing::Values(
        std::make_tuple(0, PadMode::Constant),
        std::make_tuple(10, PadMode::Constant),
        std::make_tuple(10, PadMode::Edge),
        std::make_tuple(10, PadMode::Reflect)
    ));

TEST(PadCenterTest, PadCenterTooSmallThrows) {
    ArrayXr y = ArrayXr::Ones(16);
    EXPECT_THROW(pad_center(y, 10), ParameterError);
    EXPECT_THROW(pad_center(y, 0), ParameterError);
}

// ============================================================================
// Fix Length Tests
// ============================================================================

class FixLengthTest : public ::testing::TestWithParam<int> {};

TEST_P(FixLengthTest, FixLength1D) {
    int offset = GetParam();
    ArrayXr y = ArrayXr::Ones(16);
    Eigen::Index n = offset + y.size();

    ArrayXr y_out = fix_length(y, n);

    EXPECT_EQ(y_out.size(), n);

    if (n > y.size()) {
        // Check that original data is preserved at start
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            EXPECT_NEAR(y_out(i), y(i), 1e-10);
        }
    } else {
        // Check truncation
        for (Eigen::Index i = 0; i < n; ++i) {
            EXPECT_NEAR(y_out(i), y(i), 1e-10);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(FixLengthTests, FixLengthTest,
    ::testing::Values(-5, 0, 5));

// ============================================================================
// Fix Frames Tests
// ============================================================================

TEST(FixFramesTest, BasicFixFrames) {
    std::vector<Eigen::Index> frames = {20, 35, 50, 65, 80, 95};

    auto f_fix = fix_frames(frames, 0, 120, true);

    // With padding, should include x_min at start and x_max at end
    EXPECT_EQ(f_fix.front(), 0);
    EXPECT_EQ(f_fix.back(), 120);

    // All values should be in range
    for (auto f : f_fix) {
        EXPECT_GE(f, 0);
        EXPECT_LE(f, 120);
    }
}

TEST(FixFramesTest, NegativeFramesThrow) {
    std::vector<Eigen::Index> frames = {-20, 0, 20, 40};
    EXPECT_THROW(fix_frames(frames, 0, std::nullopt, false), ParameterError);
}

// ============================================================================
// Normalize Tests
// ============================================================================

TEST(NormalizeTest, InfNorm) {
    srand_reset();
    ArrayXr x = random_array(10);
    ArrayXr x_norm = normalize(x, std::numeric_limits<Real>::infinity());

    EXPECT_EQ(x_norm.size(), x.size());
    EXPECT_NEAR(x_norm.abs().maxCoeff(), 1.0, 1e-10);
}

TEST(NormalizeTest, L1Norm) {
    srand_reset();
    ArrayXr x = random_array(10).abs();
    ArrayXr x_norm = normalize(x, 1.0);

    EXPECT_NEAR(x_norm.abs().sum(), 1.0, 1e-10);
}

TEST(NormalizeTest, L2Norm) {
    srand_reset();
    ArrayXr x = random_array(10);
    ArrayXr x_norm = normalize(x, 2.0);

    Real l2 = std::sqrt((x_norm * x_norm).sum());
    EXPECT_NEAR(l2, 1.0, 1e-10);
}

TEST(NormalizeTest, BadNormThrows) {
    ArrayXr x = ArrayXr::Ones(9);
    EXPECT_THROW(normalize(x, -0.5), ParameterError);
    EXPECT_THROW(normalize(x, -2.0), ParameterError);
}

TEST(NormalizeTest, BadInputThrows) {
    ArrayXr x = ArrayXr::Ones(9);
    x(0) = std::numeric_limits<Real>::quiet_NaN();
    EXPECT_THROW(normalize(x, std::numeric_limits<Real>::infinity()), ParameterError);
}

// ============================================================================
// Local Max/Min Tests
// ============================================================================

TEST(LocalMaxTest, LocalMax1D) {
    srand_reset();
    ArrayXr data = random_array(20);
    auto lm = localmax(data);

    EXPECT_EQ(lm.size(), data.size());

    // Verify local maxima properties
    for (Eigen::Index i = 0; i < data.size(); ++i) {
        if (lm(i)) {
            // Check it's greater than left neighbor (if exists)
            if (i > 0) {
                EXPECT_GT(data(i), data(i - 1));
            }
            // Check it's >= right neighbor (if exists)
            if (i < data.size() - 1) {
                EXPECT_GE(data(i), data(i + 1));
            }
        }
    }
}

TEST(LocalMinTest, LocalMin1D) {
    srand_reset();
    ArrayXr data = random_array(20);
    auto lm = localmin(data);

    EXPECT_EQ(lm.size(), data.size());

    // Verify local minima properties
    for (Eigen::Index i = 0; i < data.size(); ++i) {
        if (lm(i)) {
            // Check it's less than left neighbor (if exists)
            if (i > 0) {
                EXPECT_LT(data(i), data(i - 1));
            }
            // Check it's <= right neighbor (if exists)
            if (i < data.size() - 1) {
                EXPECT_LE(data(i), data(i + 1));
            }
        }
    }
}

// ============================================================================
// Peak Pick Tests
// ============================================================================

TEST(PeakPickTest, BasicPeakPick) {
    srand_reset();
    ArrayXr x = random_array(100);
    x = x.abs().square();  // Positive values

    auto peaks = peak_pick(x, 1, 1, 1, 1, 0.05, 1);

    // Check peak separation
    for (size_t i = 1; i < peaks.size(); ++i) {
        EXPECT_GT(peaks[i] - peaks[i-1], 1);
    }

    // Check each peak is a local maximum
    for (auto peak : peaks) {
        int s = std::max(0, static_cast<int>(peak) - 1);
        int e = std::min(static_cast<int>(x.size()), static_cast<int>(peak) + 1);
        Real maxVal = x.segment(s, e - s).maxCoeff();
        EXPECT_GE(x(peak), maxVal - 1e-10);
    }
}

TEST(PeakPickTest, BadParamsThrow) {
    ArrayXr x = ArrayXr::Ones(10);

    EXPECT_THROW(peak_pick(x, -1, 1, 1, 1, 0.05, 1), ParameterError);  // negative pre_max
    EXPECT_THROW(peak_pick(x, 1, -1, 1, 1, 0.05, 1), ParameterError);  // negative post_max
    EXPECT_THROW(peak_pick(x, 1, 0, 1, 1, 0.05, 1), ParameterError);   // zero post_max
    EXPECT_THROW(peak_pick(x, 1, 1, -1, 1, 0.05, 1), ParameterError);  // negative pre_avg
    EXPECT_THROW(peak_pick(x, 1, 1, 1, -1, 0.05, 1), ParameterError);  // negative post_avg
    EXPECT_THROW(peak_pick(x, 1, 1, 1, 0, 0.05, 1), ParameterError);   // zero post_avg
    EXPECT_THROW(peak_pick(x, 1, 1, 1, 1, -0.05, 1), ParameterError);  // negative delta
    EXPECT_THROW(peak_pick(x, 1, 1, 1, 1, 0.05, -1), ParameterError);  // negative wait
}

// ============================================================================
// Softmask Tests
// ============================================================================

TEST(SoftmaskTest, BasicSoftmask) {
    srand_reset();

    ArrayXXr X = random_array(100).abs().reshaped(10, 10);
    ArrayXXr X_ref = random_array(100).abs().reshaped(10, 10);

    // Zero out some rows
    X.row(3).setZero();
    X_ref.row(3).setZero();

    ArrayXXr M = softmask(X, X_ref, 1.0, false);

    // Mask should be in [0, 1]
    EXPECT_TRUE((M >= 0).all());
    EXPECT_TRUE((M <= 1).all());

    // Row 3 should be all zeros when split_zeros is false
    EXPECT_NEAR(M.row(3).sum(), 0.0, 1e-10);
}

TEST(SoftmaskTest, SplitZeros) {
    ArrayXXr X = ArrayXXr::Zero(10, 10);
    ArrayXXr X_ref = ArrayXXr::Zero(10, 10);

    ArrayXXr M = softmask(X, X_ref, 1.0, true);

    // When split_zeros is true, zeros should become 0.5
    EXPECT_TRUE((M.array() - 0.5).abs().maxCoeff() < 1e-10);
}

TEST(SoftmaskTest, Complementary) {
    ArrayXXr X = 2.0 * ArrayXXr::Ones(3, 3);
    ArrayXXr X_ref = ArrayXXr::Zero(3, 3);
    X_ref(0, 0) = 0;
    X_ref(0, 1) = 1;
    X_ref(0, 2) = 4;
    X_ref(1, 0) = 2;
    X_ref(1, 1) = 2;
    X_ref(1, 2) = 2;
    X_ref(2, 0) = 1;
    X_ref(2, 1) = 2;
    X_ref(2, 2) = 3;

    ArrayXXr M1 = softmask(X, X_ref, 1.0, false);
    ArrayXXr M2 = softmask(X_ref, X, 1.0, false);

    // M1 + M2 should equal 1 everywhere
    ArrayXXr sum = M1 + M2;
    EXPECT_TRUE((sum.array() - 1.0).abs().maxCoeff() < 1e-10);
}

TEST(SoftmaskTest, BadParamsThrow) {
    ArrayXXr ones3 = ArrayXXr::Ones(3, 3);
    ArrayXXr ones4 = ArrayXXr::Ones(4, 4);
    ArrayXXr neg_ones = -ArrayXXr::Ones(3, 3);

    EXPECT_THROW(softmask(neg_ones, ones3, 1.0, false), ParameterError);  // negative X
    EXPECT_THROW(softmask(ones3, neg_ones, 1.0, false), ParameterError);  // negative X_ref
    EXPECT_THROW(softmask(ones3, ones4, 1.0, false), ParameterError);     // shape mismatch
    EXPECT_THROW(softmask(ones3, ones3, 0.0, false), ParameterError);     // zero power
    EXPECT_THROW(softmask(ones3, ones3, -1.0, false), ParameterError);    // negative power
}

// ============================================================================
// Tiny Tests
// ============================================================================

TEST(TinyTest, FloatTiny) {
    float f = 1.0f;
    EXPECT_EQ(tiny(f), std::numeric_limits<float>::min());
}

TEST(TinyTest, DoubleTiny) {
    double d = 1.0;
    EXPECT_EQ(tiny(d), std::numeric_limits<double>::min());
}

// ============================================================================
// Valid Audio Tests
// ============================================================================

TEST(ValidAudioTest, ValidMono) {
    ArrayXr y = ArrayXr::Random(1000);
    EXPECT_TRUE(valid_audio(y));
}

TEST(ValidAudioTest, InvalidNaN) {
    ArrayXr y = ArrayXr::Random(1000);
    y(500) = std::numeric_limits<Real>::quiet_NaN();
    EXPECT_THROW(valid_audio(y), ParameterError);
}

TEST(ValidAudioTest, InvalidInf) {
    ArrayXr y = ArrayXr::Random(1000);
    y(500) = std::numeric_limits<Real>::infinity();
    EXPECT_THROW(valid_audio(y), ParameterError);
}

// ============================================================================
// Valid Int Tests
// ============================================================================

TEST(ValidIntTest, BasicValidInt) {
    EXPECT_EQ(valid_int(1.0), 1);
    EXPECT_EQ(valid_int(1.5), 1);
    EXPECT_EQ(valid_int(1.9), 1);
    EXPECT_EQ(valid_int(-1.5), -2);
}

// ============================================================================
// Is Positive Int Tests
// ============================================================================

TEST(IsPositiveIntTest, PositiveInts) {
    EXPECT_TRUE(is_positive_int(1));
    EXPECT_TRUE(is_positive_int(64));
}

TEST(IsPositiveIntTest, NonPositiveInts) {
    EXPECT_FALSE(is_positive_int(0));
    EXPECT_FALSE(is_positive_int(-1));
}

// ============================================================================
// Abs2 Tests
// ============================================================================

TEST(Abs2Test, RealAbs2) {
    ArrayXr x = ArrayXr::LinSpaced(6, -3.0, 2.0);
    ArrayXr p = abs2(x);

    for (Eigen::Index i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(p(i), x(i) * x(i), 1e-10);
    }
}

TEST(Abs2Test, ComplexAbs2) {
    ArrayXc x(3);
    x << Complex(2.0, -2.0), Complex(3.0, 0.0), Complex(0.0, 0.5);

    ArrayXr p = abs2(x);

    EXPECT_NEAR(p(0), 8.0, 1e-10);
    EXPECT_NEAR(p(1), 9.0, 1e-10);
    EXPECT_NEAR(p(2), 0.25, 1e-10);
}

// ============================================================================
// Phasor Tests
// ============================================================================

TEST(PhasorTest, BasicPhasor) {
    ArrayXr angles(2);
    angles << constants::PI / 2, -constants::PI / 3;

    ArrayXc z = phasor(angles, 1.0);

    // e^(i * pi/2) = i
    EXPECT_NEAR(z(0).real(), 0.0, 1e-10);
    EXPECT_NEAR(z(0).imag(), 1.0, 1e-10);

    // e^(i * -pi/3) = cos(-pi/3) + i*sin(-pi/3) = 0.5 - i*sqrt(3)/2
    EXPECT_NEAR(z(1).real(), 0.5, 1e-10);
    EXPECT_NEAR(z(1).imag(), -std::sqrt(3.0) / 2, 1e-10);
}

TEST(PhasorTest, PhasorWithMagnitude) {
    ArrayXr angles(1);
    angles << constants::PI / 2;

    ArrayXc z = phasor(angles, 2.0);

    EXPECT_NEAR(z(0).real(), 0.0, 1e-10);
    EXPECT_NEAR(z(0).imag(), 2.0, 1e-10);
}

// ============================================================================
// Fill Off Diagonal Tests
// ============================================================================

TEST(FillOffDiagonalTest, Square8x8) {
    ArrayXXr x = ArrayXXr::Ones(8, 8);
    ArrayXXr result = fill_off_diagonal(x, 0.0, false);

    // Check diagonal band is preserved
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(result(i, i), 1.0, 1e-10);
    }
}

// ============================================================================
// Stack Tests
// ============================================================================

TEST(StackTest, BasicStack) {
    ArrayXr x1 = ArrayXr::Ones(3);
    ArrayXr x2 = -ArrayXr::Ones(3);

    std::vector<ArrayXr> arrays = {x1, x2};
    ArrayXXr xs = stack(arrays, 0);

    EXPECT_EQ(xs.rows(), 2);
    EXPECT_EQ(xs.cols(), 3);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(xs(0, i), 1.0, 1e-10);
        EXPECT_NEAR(xs(1, i), -1.0, 1e-10);
    }
}

TEST(StackTest, EmptyStackThrows) {
    std::vector<ArrayXr> arrays;
    EXPECT_THROW(stack(arrays, 0), ParameterError);
}

TEST(StackTest, MismatchedShapesThrow) {
    ArrayXr x1 = ArrayXr::Ones(3);
    ArrayXr x2 = ArrayXr::Ones(2);

    std::vector<ArrayXr> arrays = {x1, x2};
    EXPECT_THROW(stack(arrays, 0), ParameterError);
}
