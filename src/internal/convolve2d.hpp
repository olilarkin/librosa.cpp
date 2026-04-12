#pragma once

#include <librosa/types.hpp>
#include "pad2d.hpp"

namespace librosa {
namespace internal {

/// 2D convolution with reflect padding (matches scipy.ndimage.convolve default)
/// @param input Input 2D array
/// @param kernel Convolution kernel
/// @param mode Padding mode ("reflect", "constant", "edge")
/// @return Convolved array (same size as input)
inline ArrayXXr convolve2d(
    const ArrayXXr& input,
    const ArrayXXr& kernel,
    const std::string& mode = "reflect") {

    Eigen::Index krows = kernel.rows();
    Eigen::Index kcols = kernel.cols();
    int pad_rows = static_cast<int>(krows / 2);
    int pad_cols = static_cast<int>(kcols / 2);

    // Pad the input
    ArrayXXr padded = pad_array_2d(input, pad_rows, pad_cols, mode);

    // Output array
    ArrayXXr result(input.rows(), input.cols());

    // Convolve
    for (Eigen::Index i = 0; i < input.rows(); ++i) {
        for (Eigen::Index j = 0; j < input.cols(); ++j) {
            Real sum = 0.0;
            for (Eigen::Index ki = 0; ki < krows; ++ki) {
                for (Eigen::Index kj = 0; kj < kcols; ++kj) {
                    sum += padded(i + ki, j + kj) * kernel(ki, kj);
                }
            }
            result(i, j) = sum;
        }
    }

    return result;
}

} // namespace internal
} // namespace librosa
