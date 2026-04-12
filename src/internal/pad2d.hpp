#pragma once

#include <librosa/types.hpp>
#include <librosa/util/exceptions.hpp>
#include <algorithm>
#include <string>

namespace librosa {
namespace internal {

/// Pad a 2D array with specified mode
/// @param S Input array
/// @param pad_rows Rows of padding on each side
/// @param pad_cols Columns of padding on each side
/// @param mode "reflect", "constant", or "edge"
/// @return Padded array
inline ArrayXXr pad_array_2d(
    const ArrayXXr& S,
    int pad_rows,
    int pad_cols,
    const std::string& mode) {

    Eigen::Index rows = S.rows();
    Eigen::Index cols = S.cols();
    Eigen::Index new_rows = rows + 2 * pad_rows;
    Eigen::Index new_cols = cols + 2 * pad_cols;

    ArrayXXr padded = ArrayXXr::Zero(new_rows, new_cols);

    // Copy center
    padded.block(pad_rows, pad_cols, rows, cols) = S;

    if (mode == "constant") {
        // Already zero-padded
    } else if (mode == "edge") {
        // Top edge
        for (int i = 0; i < pad_rows; ++i) {
            padded.row(i).segment(pad_cols, cols) = S.row(0);
        }
        // Bottom edge
        for (int i = 0; i < pad_rows; ++i) {
            padded.row(new_rows - 1 - i).segment(pad_cols, cols) = S.row(rows - 1);
        }
        // Left edge
        for (int j = 0; j < pad_cols; ++j) {
            padded.col(j).segment(pad_rows, rows) = S.col(0);
        }
        // Right edge
        for (int j = 0; j < pad_cols; ++j) {
            padded.col(new_cols - 1 - j).segment(pad_rows, rows) = S.col(cols - 1);
        }
        // Corners
        for (int i = 0; i < pad_rows; ++i) {
            for (int j = 0; j < pad_cols; ++j) {
                padded(i, j) = S(0, 0);
                padded(i, new_cols - 1 - j) = S(0, cols - 1);
                padded(new_rows - 1 - i, j) = S(rows - 1, 0);
                padded(new_rows - 1 - i, new_cols - 1 - j) = S(rows - 1, cols - 1);
            }
        }
    } else if (mode == "reflect") {
        // Top edge (reflect without edge)
        for (int i = 0; i < pad_rows; ++i) {
            int src_row = std::min(i + 1, static_cast<int>(rows - 1));
            padded.row(pad_rows - 1 - i).segment(pad_cols, cols) = S.row(src_row);
        }
        // Bottom edge
        for (int i = 0; i < pad_rows; ++i) {
            int src_row = std::max(0, static_cast<int>(rows - 2 - i));
            padded.row(pad_rows + rows + i).segment(pad_cols, cols) = S.row(src_row);
        }
        // Left edge
        for (int j = 0; j < pad_cols; ++j) {
            int src_col = std::min(j + 1, static_cast<int>(cols - 1));
            padded.col(pad_cols - 1 - j).segment(pad_rows, rows) = S.col(src_col);
        }
        // Right edge
        for (int j = 0; j < pad_cols; ++j) {
            int src_col = std::max(0, static_cast<int>(cols - 2 - j));
            padded.col(pad_cols + cols + j).segment(pad_rows, rows) = S.col(src_col);
        }
        // Corners - reflect in both directions
        for (int i = 0; i < pad_rows; ++i) {
            for (int j = 0; j < pad_cols; ++j) {
                int src_row_top = std::min(i + 1, static_cast<int>(rows - 1));
                int src_row_bot = std::max(0, static_cast<int>(rows - 2 - i));
                int src_col_left = std::min(j + 1, static_cast<int>(cols - 1));
                int src_col_right = std::max(0, static_cast<int>(cols - 2 - j));

                padded(pad_rows - 1 - i, pad_cols - 1 - j) = S(src_row_top, src_col_left);
                padded(pad_rows - 1 - i, pad_cols + cols + j) = S(src_row_top, src_col_right);
                padded(pad_rows + rows + i, pad_cols - 1 - j) = S(src_row_bot, src_col_left);
                padded(pad_rows + rows + i, pad_cols + cols + j) = S(src_row_bot, src_col_right);
            }
        }
    } else {
        throw ParameterError("Unknown padding mode: " + mode);
    }

    return padded;
}

} // namespace internal
} // namespace librosa
