#include <librosa/feature/utils.hpp>
#include <librosa/util/exceptions.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <string>

namespace librosa {
namespace feature {

namespace {

/// Compute Savitzky-Golay filter coefficients for the given derivative order
/// @param width Window width (odd)
/// @param polyorder Polynomial order for fitting
/// @param deriv Derivative order
/// @return Filter coefficients of length width
ArrayXr savgol_coeffs(int width, int polyorder, int deriv) {
    int half_w = width / 2;

    // Build Vandermonde matrix: V[i,j] = x_i^j where x_i = i - half_w
    Eigen::MatrixXd V(width, polyorder + 1);
    for (int i = 0; i < width; ++i) {
        double x = static_cast<double>(i - half_w);
        double xp = 1.0;
        for (int j = 0; j <= polyorder; ++j) {
            V(i, j) = xp;
            xp *= x;
        }
    }

    // Compute (V^T V)^{-1} V^T — the pseudoinverse
    Eigen::MatrixXd VtV = V.transpose() * V;
    Eigen::MatrixXd VtV_inv = VtV.inverse();
    Eigen::MatrixXd proj = VtV_inv * V.transpose();

    // The derivative coefficients are row `deriv` of proj, scaled by deriv!
    double factorial = 1.0;
    for (int d = 1; d <= deriv; ++d) {
        factorial *= d;
    }

    ArrayXr coeffs(width);
    for (int i = 0; i < width; ++i) {
        coeffs(i) = static_cast<Real>(proj(deriv, i) * factorial);
    }

    return coeffs;
}

/// Pad a 1D array according to the specified mode
/// @param x Input array
/// @param pad Number of samples to pad on each side
/// @param mode Padding mode
/// @return Padded array
ArrayXr pad_1d(const ArrayXr& x, int pad, const std::string& mode) {
    Eigen::Index n = x.size();
    ArrayXr padded(n + 2 * pad);

    // Copy center
    padded.segment(pad, n) = x;

    if (mode == "constant") {
        padded.head(pad).setZero();
        padded.tail(pad).setZero();
    } else if (mode == "nearest" || mode == "edge") {
        for (int i = 0; i < pad; ++i) {
            padded(i) = x(0);
            padded(pad + n + i) = x(n - 1);
        }
    } else if (mode == "mirror") {
        // Mirror without repeating edge: a b c d | d c b a | a b c d
        // scipy interp mode actually pads differently but for other modes:
        // mirror = reflect without edge point
        for (int i = 0; i < pad; ++i) {
            int src = std::min(static_cast<int>(n - 1), i + 1);
            padded(pad - 1 - i) = x(src);
        }
        for (int i = 0; i < pad; ++i) {
            int src = std::max(0, static_cast<int>(n - 2 - i));
            padded(pad + n + i) = x(src);
        }
    } else if (mode == "wrap") {
        for (int i = 0; i < pad; ++i) {
            padded(pad - 1 - i) = x((n - 1 - i) % n);
            padded(pad + n + i) = x(i % n);
        }
    } else {
        throw ParameterError("Unknown padding mode: " + mode);
    }

    return padded;
}

/// Apply Savitzky-Golay filter to a 1D signal with interp mode
/// For interp mode, edge values are computed by fitting a polynomial
/// to the available data points and evaluating the derivative.
ArrayXr apply_savgol_interp(const ArrayXr& x, const ArrayXr& coeffs, int width,
                            int polyorder, int deriv) {
    Eigen::Index n = x.size();
    int half_w = width / 2;
    ArrayXr result(n);

    // Interior: standard convolution
    for (Eigen::Index i = half_w; i < n - half_w; ++i) {
        Real sum = 0.0;
        for (int k = 0; k < width; ++k) {
            sum += coeffs(k) * x(i - half_w + k);
        }
        result(i) = sum;
    }

    // Edge handling: fit polynomial to first/last width points
    double factorial = 1.0;
    for (int d = 1; d <= deriv; ++d) {
        factorial *= d;
    }

    // Left edge: fit polynomial to first `width` points
    {
        Eigen::MatrixXd V(width, polyorder + 1);
        for (int i = 0; i < width; ++i) {
            double xi = static_cast<double>(i);
            double xp = 1.0;
            for (int j = 0; j <= polyorder; ++j) {
                V(i, j) = xp;
                xp *= xi;
            }
        }
        Eigen::VectorXd y_vals(width);
        for (int i = 0; i < width; ++i) {
            y_vals(i) = x(i);
        }
        // Least squares solve
        Eigen::VectorXd poly_coeffs = V.colPivHouseholderQr().solve(y_vals);

        // Evaluate derivative at each edge point
        for (int i = 0; i < half_w; ++i) {
            double xi = static_cast<double>(i);
            double val = 0.0;
            // d-th derivative of polynomial sum(c_j * x^j) at x=xi
            for (int j = deriv; j <= polyorder; ++j) {
                double coeff = poly_coeffs(j);
                double factor = 1.0;
                for (int d = 0; d < deriv; ++d) {
                    factor *= (j - d);
                }
                val += coeff * factor * std::pow(xi, j - deriv);
            }
            result(i) = static_cast<Real>(val);
        }
    }

    // Right edge: fit polynomial to last `width` points
    {
        Eigen::MatrixXd V(width, polyorder + 1);
        for (int i = 0; i < width; ++i) {
            double xi = static_cast<double>(i);
            double xp = 1.0;
            for (int j = 0; j <= polyorder; ++j) {
                V(i, j) = xp;
                xp *= xi;
            }
        }
        Eigen::VectorXd y_vals(width);
        Eigen::Index start = n - width;
        for (int i = 0; i < width; ++i) {
            y_vals(i) = x(start + i);
        }
        Eigen::VectorXd poly_coeffs = V.colPivHouseholderQr().solve(y_vals);

        for (Eigen::Index i = n - half_w; i < n; ++i) {
            double xi = static_cast<double>(i - start);
            double val = 0.0;
            for (int j = deriv; j <= polyorder; ++j) {
                double coeff = poly_coeffs(j);
                double factor = 1.0;
                for (int d = 0; d < deriv; ++d) {
                    factor *= (j - d);
                }
                val += coeff * factor * std::pow(xi, j - deriv);
            }
            result(i) = static_cast<Real>(val);
        }
    }

    return result;
}

/// Apply Savitzky-Golay filter to a 1D signal with padding mode
ArrayXr apply_savgol_padded(const ArrayXr& x, const ArrayXr& coeffs, int width,
                            const std::string& mode) {
    int half_w = width / 2;
    ArrayXr padded = pad_1d(x, half_w, mode);

    Eigen::Index n = x.size();
    ArrayXr result(n);

    for (Eigen::Index i = 0; i < n; ++i) {
        Real sum = 0.0;
        for (int k = 0; k < width; ++k) {
            sum += coeffs(k) * padded(i + k);
        }
        result(i) = sum;
    }

    return result;
}

} // anonymous namespace

ArrayXXr delta(const ArrayXXr& data, int width, int order,
               int axis, const std::string& mode) {
    if (width < 3 || width % 2 != 1) {
        throw ParameterError("width must be an odd integer >= 3");
    }
    if (order <= 0) {
        throw ParameterError("order must be a positive integer");
    }

    // Resolve axis
    int real_axis = axis;
    if (real_axis < 0) {
        real_axis = 1;  // For 2D arrays, -1 means columns (time axis = axis 1)
    }

    Eigen::Index n_along = (real_axis == 0) ? data.rows() : data.cols();

    if (mode == "interp" && width > n_along) {
        throw ParameterError("when mode='interp', width cannot exceed data.shape[axis]");
    }

    int polyorder = order;  // Match scipy default: polyorder = order

    // Compute SG coefficients
    ArrayXr coeffs = savgol_coeffs(width, polyorder, order);

    ArrayXXr result(data.rows(), data.cols());

    if (real_axis == 1) {
        // Apply along columns (time axis) — process each row
        for (Eigen::Index r = 0; r < data.rows(); ++r) {
            ArrayXr row = data.row(r);
            if (mode == "interp") {
                result.row(r) = apply_savgol_interp(row, coeffs, width, polyorder, order);
            } else {
                result.row(r) = apply_savgol_padded(row, coeffs, width, mode);
            }
        }
    } else {
        // Apply along rows (feature axis) — process each column
        for (Eigen::Index c = 0; c < data.cols(); ++c) {
            ArrayXr col = data.col(c);
            if (mode == "interp") {
                result.col(c) = apply_savgol_interp(col, coeffs, width, polyorder, order);
            } else {
                result.col(c) = apply_savgol_padded(col, coeffs, width, mode);
            }
        }
    }

    return result;
}

ArrayXXr stack_memory(const ArrayXXr& data, int n_steps, int delay,
                      PadMode mode) {
    if (n_steps < 1) {
        throw ParameterError("n_steps must be a positive integer");
    }
    if (delay == 0) {
        throw ParameterError("delay must be a non-zero integer");
    }

    Eigen::Index n_features = data.rows();
    Eigen::Index t = data.cols();

    if (n_steps == 1) {
        return data;
    }

    // Compute padding amount
    int pad_amount = std::abs(delay) * (n_steps - 1);

    // Create padded data along time axis
    ArrayXXr padded;
    if (delay > 0) {
        // Pad on the left (history)
        padded.resize(n_features, pad_amount + t);

        if (mode == PadMode::Constant) {
            padded.leftCols(pad_amount).setZero();
        } else if (mode == PadMode::Edge) {
            for (int i = 0; i < pad_amount; ++i) {
                padded.col(i) = data.col(0);
            }
        } else if (mode == PadMode::Reflect) {
            for (int i = 0; i < pad_amount; ++i) {
                Eigen::Index src_idx = pad_amount - i;
                if (src_idx >= t) src_idx = t - 1;
                padded.col(i) = data.col(src_idx);
            }
        } else {
            // Default: constant (zero) padding
            padded.leftCols(pad_amount).setZero();
        }

        padded.rightCols(t) = data;
    } else {
        // Pad on the right (future)
        padded.resize(n_features, t + pad_amount);
        padded.leftCols(t) = data;

        if (mode == PadMode::Constant) {
            padded.rightCols(pad_amount).setZero();
        } else if (mode == PadMode::Edge) {
            for (int i = 0; i < pad_amount; ++i) {
                padded.col(t + i) = data.col(t - 1);
            }
        } else if (mode == PadMode::Reflect) {
            for (int i = 0; i < pad_amount; ++i) {
                Eigen::Index src_idx = t - 2 - i;
                if (src_idx < 0) src_idx = 0;
                padded.col(t + i) = data.col(src_idx);
            }
        } else {
            padded.rightCols(pad_amount).setZero();
        }
    }

    // Build output: stack n_steps copies with shifted indexing
    ArrayXXr result(n_features * n_steps, t);

    for (int step = 0; step < n_steps; ++step) {
        Eigen::Index offset;
        if (delay > 0) {
            // For positive delay: step 0 = current, step 1 = delay frames back, etc.
            offset = pad_amount - step * delay;
        } else {
            // For negative delay: step 0 = current, step 1 = |delay| frames forward, etc.
            offset = step * std::abs(delay);
        }

        for (Eigen::Index col = 0; col < t; ++col) {
            result.block(step * n_features, col, n_features, 1) =
                padded.col(offset + col);
        }
    }

    return result;
}

} // namespace feature
} // namespace librosa
