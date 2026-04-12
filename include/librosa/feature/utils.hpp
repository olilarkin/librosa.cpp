#pragma once

#include "../types.hpp"

namespace librosa {
namespace feature {

/// Compute delta features using Savitzky-Golay filtering
///
/// Computes the local estimate of the derivative of the input data
/// along the selected axis.
///
/// @param data Input feature matrix (e.g., spectrogram)
/// @param width Number of frames over which to compute the delta features (odd, >= 3)
/// @param order Order of the difference operator (1 for first derivative, 2 for second, etc.)
/// @param axis Axis along which to compute deltas (-1 for columns/time axis)
/// @param mode Padding mode: "interp", "nearest", "mirror", "constant", "wrap"
/// @return Delta matrix of data at specified order
ArrayXXr delta(const ArrayXXr& data, int width = 9, int order = 1,
               int axis = -1, const std::string& mode = "interp");

/// Stack short-term history of a feature matrix
///
/// Creates a vertically stacked version of the input data by
/// stacking consecutive frames with a given delay.
///
/// @param data Input feature matrix [shape: (n_features, t)]
/// @param n_steps Number of steps to stack (must be >= 1)
/// @param delay Number of frames to delay between steps
///        Positive delay stacks history (pad left), negative stacks future (pad right)
/// @param mode Padding mode for out-of-bounds frames
/// @return Stacked feature matrix [shape: (n_features * n_steps, t)]
ArrayXXr stack_memory(const ArrayXXr& data, int n_steps = 2, int delay = 1,
                      PadMode mode = PadMode::Constant);

} // namespace feature
} // namespace librosa
