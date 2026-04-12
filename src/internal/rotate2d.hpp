#pragma once

#include <librosa/types.hpp>
#include <cmath>

namespace librosa {
namespace internal {

/// Rotate a 2D image by a given angle around its center using bilinear interpolation.
/// Out-of-bounds pixels are filled with 0.
/// @param input Input 2D array
/// @param angle Rotation angle in radians (positive = counter-clockwise)
/// @return Rotated array (same size as input)
inline ArrayXXr rotate2d(const ArrayXXr& input, Real angle) {
    Eigen::Index rows = input.rows();
    Eigen::Index cols = input.cols();

    ArrayXXr result = ArrayXXr::Zero(rows, cols);

    Real cy = (rows - 1) / 2.0;
    Real cx = (cols - 1) / 2.0;

    Real cos_a = std::cos(angle);
    Real sin_a = std::sin(angle);

    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            // Map destination (i, j) back to source coordinates
            Real di = static_cast<Real>(i) - cy;
            Real dj = static_cast<Real>(j) - cx;

            Real si = cos_a * di + sin_a * dj + cy;
            Real sj = -sin_a * di + cos_a * dj + cx;

            // Bilinear interpolation
            int si0 = static_cast<int>(std::floor(si));
            int sj0 = static_cast<int>(std::floor(sj));
            int si1 = si0 + 1;
            int sj1 = sj0 + 1;

            if (si0 < 0 || si1 >= rows || sj0 < 0 || sj1 >= cols) {
                result(i, j) = 0.0;
                continue;
            }

            Real fi = si - si0;
            Real fj = sj - sj0;

            result(i, j) = (1.0 - fi) * (1.0 - fj) * input(si0, sj0)
                         + (1.0 - fi) * fj         * input(si0, sj1)
                         + fi         * (1.0 - fj) * input(si1, sj0)
                         + fi         * fj         * input(si1, sj1);
        }
    }

    return result;
}

} // namespace internal
} // namespace librosa
