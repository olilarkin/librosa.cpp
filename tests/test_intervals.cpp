#include <gtest/gtest.h>
#include <librosa/core/intervals.hpp>
#include <librosa/util/exceptions.hpp>
#include <cmath>

using namespace librosa;

// ============================================================================
// Pythagorean Intervals Tests
// ============================================================================

TEST(PythagoreanTest, Twelve) {
    auto intervals = pythagorean_intervals(12);
    ASSERT_EQ(intervals.size(), 12);

    // First should be 1.0 (unison)
    EXPECT_NEAR(intervals(0), 1.0, 1e-6);

    // All should be in [1, 2)
    for (int i = 0; i < 12; ++i) {
        EXPECT_GE(intervals(i), 1.0);
        EXPECT_LT(intervals(i), 2.0);
    }

    // Should be sorted
    for (int i = 1; i < 12; ++i) {
        EXPECT_GT(intervals(i), intervals(i - 1));
    }

    // Perfect fifth (3/2 = 1.5) should be present
    bool has_fifth = false;
    for (int i = 0; i < 12; ++i) {
        if (std::abs(intervals(i) - 1.5) < 0.001) {
            has_fifth = true;
            break;
        }
    }
    EXPECT_TRUE(has_fifth);
}

TEST(PythagoreanTest, SevenUnsorted) {
    auto intervals = pythagorean_intervals(7, false);
    ASSERT_EQ(intervals.size(), 7);

    // First should be 1.0
    EXPECT_NEAR(intervals(0), 1.0, 1e-6);
    // Second should be 3/2 = 1.5 (one fifth up)
    EXPECT_NEAR(intervals(1), 1.5, 1e-6);
    // Third should be 9/8 = 1.125 (two fifths up, folded)
    EXPECT_NEAR(intervals(2), 1.125, 1e-6);
}

// ============================================================================
// P-limit Intervals Tests
// ============================================================================

TEST(PlimitTest, ThreeLimit) {
    // 3-limit should be similar but not identical to Pythagorean
    auto intervals = plimit_intervals({3}, 12);
    ASSERT_EQ(intervals.size(), 12);

    // All in [1, 2)
    for (int i = 0; i < 12; ++i) {
        EXPECT_GE(intervals(i), 1.0);
        EXPECT_LT(intervals(i), 2.0);
    }

    // First should be 1.0
    EXPECT_NEAR(intervals(0), 1.0, 1e-6);
}

TEST(PlimitTest, FiveLimit) {
    auto intervals = plimit_intervals({3, 5}, 7);
    ASSERT_EQ(intervals.size(), 7);

    // First should be 1.0
    EXPECT_NEAR(intervals(0), 1.0, 1e-6);

    // 3/2 = 1.5 (perfect fifth) should appear
    bool has_fifth = false;
    for (int i = 0; i < 7; ++i) {
        if (std::abs(intervals(i) - 1.5) < 0.001) {
            has_fifth = true;
            break;
        }
    }
    EXPECT_TRUE(has_fifth);

    // 5/4 = 1.25 (major third) should appear
    bool has_third = false;
    for (int i = 0; i < 7; ++i) {
        if (std::abs(intervals(i) - 1.25) < 0.001) {
            has_third = true;
            break;
        }
    }
    EXPECT_TRUE(has_third);
}

TEST(PlimitTest, SevenLimit) {
    auto intervals = plimit_intervals({3, 5, 7}, 12);
    ASSERT_EQ(intervals.size(), 12);

    EXPECT_NEAR(intervals(0), 1.0, 1e-6);
    for (int i = 1; i < 12; ++i) {
        EXPECT_GT(intervals(i), intervals(i - 1));
    }
}

// ============================================================================
// Interval Frequencies Tests
// ============================================================================

TEST(IntervalFreqTest, EqualTemperament) {
    auto freqs = interval_frequencies(12, 440.0, "equal");
    ASSERT_EQ(freqs.size(), 12);

    EXPECT_NEAR(freqs(0), 440.0, 0.01);

    // 12th semitone should be close to 880 Hz (octave)
    EXPECT_NEAR(freqs(11), 440.0 * std::pow(2.0, 11.0 / 12.0), 0.01);
}

TEST(IntervalFreqTest, Pythagorean) {
    auto freqs = interval_frequencies(24, 55.0, "pythagorean", 12);
    ASSERT_EQ(freqs.size(), 24);

    // First should be fmin
    EXPECT_NEAR(freqs(0), 55.0, 0.01);

    // All should be positive and increasing
    for (int i = 1; i < 24; ++i) {
        EXPECT_GT(freqs(i), freqs(i - 1));
    }
}

TEST(IntervalFreqTest, JI5) {
    auto freqs = interval_frequencies(24, 55.0, "ji5", 12);
    ASSERT_EQ(freqs.size(), 24);
    EXPECT_NEAR(freqs(0), 55.0, 0.01);
}

TEST(IntervalFreqTest, CustomIntervals) {
    ArrayXr custom(3);
    custom << 1.0, 4.0/3.0, 3.0/2.0;

    auto freqs = interval_frequencies(9, 55.0, custom);
    ASSERT_EQ(freqs.size(), 9);

    EXPECT_NEAR(freqs(0), 55.0, 0.01);
    EXPECT_NEAR(freqs(1), 55.0 * 4.0 / 3.0, 0.01);
    EXPECT_NEAR(freqs(2), 55.0 * 3.0 / 2.0, 0.01);
    // Second octave
    EXPECT_NEAR(freqs(3), 110.0, 0.01);
}

TEST(IntervalFreqTest, UnknownTypeThrows) {
    EXPECT_THROW(interval_frequencies(12, 440.0, "unknown"), ParameterError);
}
