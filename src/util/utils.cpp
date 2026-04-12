#include "librosa/util/utils.hpp"
#include <cmath>
#include <set>
#include <numeric>

namespace librosa {
namespace util {

bool valid_audio(const ArrayXr& y) {
    if (y.size() == 0) {
        throw ParameterError("Audio data must be at least one-dimensional");
    }
    if (!y.isFinite().all()) {
        throw ParameterError("Audio buffer is not finite everywhere");
    }
    return true;
}

bool valid_audio(const ArrayXXr& y) {
    if (y.rows() == 0 || y.cols() == 0) {
        throw ParameterError("Audio data must be at least one-dimensional");
    }
    if (!y.isFinite().all()) {
        throw ParameterError("Audio buffer is not finite everywhere");
    }
    return true;
}

bool is_positive_int(int x) {
    return x > 0;
}

int valid_int(double x, bool use_floor) {
    if (use_floor) {
        return static_cast<int>(std::floor(x));
    }
    return static_cast<int>(x);
}

bool valid_intervals(const ArrayXXr& intervals) {
    if (intervals.cols() != 2) {
        throw ParameterError("intervals must have shape (n, 2)");
    }
    for (Eigen::Index i = 0; i < intervals.rows(); ++i) {
        if (intervals(i, 0) > intervals(i, 1)) {
            throw ParameterError("intervals must have non-negative durations");
        }
    }
    return true;
}

ArrayXr pad_center(const ArrayXr& data, Eigen::Index size,
                   PadMode mode, Real constant_value) {
    Eigen::Index n = data.size();
    if (size < n) {
        throw ParameterError("Target size must be at least input size");
    }

    Eigen::Index lpad = (size - n) / 2;
    Eigen::Index rpad = size - n - lpad;

    ArrayXr result(size);

    // Fill with padding
    switch (mode) {
        case PadMode::Constant:
            result.setConstant(constant_value);
            break;
        case PadMode::Edge:
            result.head(lpad).setConstant(data(0));
            result.tail(rpad).setConstant(data(n - 1));
            break;
        case PadMode::Reflect:
            for (Eigen::Index i = 0; i < lpad; ++i) {
                Eigen::Index idx = (lpad - i) % n;
                result(i) = data(idx);
            }
            for (Eigen::Index i = 0; i < rpad; ++i) {
                Eigen::Index idx = (n - 2 - (i % (n - 1)));
                if (idx < 0) idx = -idx;
                result(lpad + n + i) = data(idx);
            }
            break;
        default:
            result.setConstant(constant_value);
            break;
    }

    // Copy data to center
    result.segment(lpad, n) = data;

    return result;
}

ArrayXXr pad_center(const ArrayXXr& data, Eigen::Index size, int axis,
                    PadMode mode, Real constant_value) {
    // Normalize axis
    if (axis < 0) {
        axis = 2 + axis;  // For 2D array
    }

    if (axis == 0) {
        // Pad along rows
        Eigen::Index n = data.rows();
        if (size < n) {
            throw ParameterError("Target size must be at least input size");
        }
        Eigen::Index lpad = (size - n) / 2;

        ArrayXXr result(size, data.cols());
        result.setConstant(constant_value);

        result.block(lpad, 0, n, data.cols()) = data;
        return result;
    } else {
        // Pad along columns
        Eigen::Index n = data.cols();
        if (size < n) {
            throw ParameterError("Target size must be at least input size");
        }
        Eigen::Index lpad = (size - n) / 2;

        ArrayXXr result(data.rows(), size);
        result.setConstant(constant_value);

        result.block(0, lpad, data.rows(), n) = data;
        return result;
    }
}

ArrayXr fix_length(const ArrayXr& data, Eigen::Index size, PadMode mode) {
    Eigen::Index n = data.size();

    if (n > size) {
        // Trim
        return data.head(size);
    } else if (n < size) {
        // Pad
        ArrayXr result(size);
        result.head(n) = data;

        switch (mode) {
            case PadMode::Constant:
                result.tail(size - n).setZero();
                break;
            case PadMode::Edge:
                result.tail(size - n).setConstant(data(n - 1));
                break;
            default:
                result.tail(size - n).setZero();
                break;
        }
        return result;
    }
    return data;
}

ArrayXXr fix_length(const ArrayXXr& data, Eigen::Index size, int axis, PadMode mode) {
    if (axis < 0) {
        axis = 2 + axis;
    }

    if (axis == 0) {
        Eigen::Index n = data.rows();
        if (n > size) {
            return data.topRows(size);
        } else if (n < size) {
            ArrayXXr result(size, data.cols());
            result.topRows(n) = data;
            result.bottomRows(size - n).setZero();
            return result;
        }
    } else {
        Eigen::Index n = data.cols();
        if (n > size) {
            return data.leftCols(size);
        } else if (n < size) {
            ArrayXXr result(data.rows(), size);
            result.leftCols(n) = data;
            result.rightCols(size - n).setZero();
            return result;
        }
    }
    return data;
}

std::vector<Eigen::Index> fix_frames(const std::vector<Eigen::Index>& frames,
                                      Eigen::Index x_min,
                                      std::optional<Eigen::Index> x_max,
                                      bool pad) {
    std::set<Eigen::Index> frame_set;

    for (auto f : frames) {
        if (f < 0) {
            throw ParameterError("Negative frame index detected");
        }
        if (f >= x_min && (!x_max || f <= *x_max)) {
            frame_set.insert(f);
        }
    }

    if (pad) {
        frame_set.insert(x_min);
        if (x_max) {
            frame_set.insert(*x_max);
        }
    }

    return std::vector<Eigen::Index>(frame_set.begin(), frame_set.end());
}

ArrayXXr frame(const ArrayXr& x, Eigen::Index frame_length, Eigen::Index hop_length) {
    if (x.size() < frame_length) {
        throw ParameterError("Input is too short for frame_length");
    }
    if (hop_length < 1) {
        throw ParameterError("Invalid hop_length");
    }

    Eigen::Index n_frames = 1 + (x.size() - frame_length) / hop_length;
    ArrayXXr result(frame_length, n_frames);

    for (Eigen::Index i = 0; i < n_frames; ++i) {
        result.col(i) = x.segment(i * hop_length, frame_length);
    }

    return result;
}

ArrayXXr normalize(const ArrayXXr& S, Real norm, int axis,
                   std::optional<Real> threshold,
                   std::optional<bool> fill) {
    Real thresh = threshold.value_or(tiny<Real>());

    if (thresh <= 0) {
        throw ParameterError("threshold must be strictly positive");
    }

    if (!S.isFinite().all()) {
        throw ParameterError("Input must be finite");
    }

    ArrayXXr mag = S.abs();
    ArrayXXr length;
    Real fill_norm = 1.0;

    if (std::isinf(norm) && norm > 0) {
        // L-infinity norm (max)
        if (axis == 0) {
            length = mag.colwise().maxCoeff().replicate(S.rows(), 1);
        } else {
            length = mag.rowwise().maxCoeff().replicate(1, S.cols());
        }
    } else if (std::isinf(norm) && norm < 0) {
        // -infinity norm (min)
        if (axis == 0) {
            length = mag.colwise().minCoeff().replicate(S.rows(), 1);
        } else {
            length = mag.rowwise().minCoeff().replicate(1, S.cols());
        }
    } else if (norm == 0) {
        // L0 "norm" (count of non-zeros)
        if (fill && *fill) {
            throw ParameterError("Cannot normalize with norm=0 and fill=True");
        }
        ArrayXXr nonzero = (mag > 0).cast<Real>();
        if (axis == 0) {
            length = nonzero.colwise().sum().replicate(S.rows(), 1);
        } else {
            length = nonzero.rowwise().sum().replicate(1, S.cols());
        }
    } else if (norm > 0) {
        // Lp norm
        if (axis == 0) {
            length = mag.pow(norm).colwise().sum().pow(1.0 / norm).replicate(S.rows(), 1);
            fill_norm = std::pow(S.rows(), -1.0 / norm);
        } else {
            length = mag.pow(norm).rowwise().sum().pow(1.0 / norm).replicate(1, S.cols());
            fill_norm = std::pow(S.cols(), -1.0 / norm);
        }
    } else {
        throw ParameterError("Unsupported norm");
    }

    ArrayXXr result = ArrayXXr::Zero(S.rows(), S.cols());

    // Handle thresholding
    for (Eigen::Index i = 0; i < S.rows(); ++i) {
        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            if (length(i, j) >= thresh) {
                result(i, j) = S(i, j) / length(i, j);
            } else if (!fill) {
                result(i, j) = S(i, j);  // Leave unchanged
            } else if (*fill) {
                result(i, j) = fill_norm;  // Fill with unit norm value
            }
            // else fill=false: leave as zero
        }
    }

    return result;
}

ArrayXr normalize(const ArrayXr& S, Real norm,
                  std::optional<Real> threshold,
                  std::optional<bool> fill) {
    Real thresh = threshold.value_or(tiny<Real>());

    if (thresh <= 0) {
        throw ParameterError("threshold must be strictly positive");
    }

    if (!S.isFinite().all()) {
        throw ParameterError("Input must be finite");
    }

    ArrayXr mag = S.abs();
    Real length;
    Real fill_norm = 1.0;

    if (std::isinf(norm) && norm > 0) {
        length = mag.maxCoeff();
    } else if (std::isinf(norm) && norm < 0) {
        length = mag.minCoeff();
    } else if (norm == 0) {
        if (fill && *fill) {
            throw ParameterError("Cannot normalize with norm=0 and fill=True");
        }
        length = (mag > 0).cast<Real>().sum();
    } else if (norm > 0) {
        length = std::pow(mag.pow(norm).sum(), 1.0 / norm);
        fill_norm = std::pow(S.size(), -1.0 / norm);
    } else {
        throw ParameterError("Unsupported norm");
    }

    if (length >= thresh) {
        return S / length;
    } else if (!fill) {
        return S;
    } else if (*fill) {
        return ArrayXr::Constant(S.size(), fill_norm);
    }
    return ArrayXr::Zero(S.size());
}

Eigen::Array<bool, Eigen::Dynamic, 1> localmax(const ArrayXr& x) {
    Eigen::Array<bool, Eigen::Dynamic, 1> result(x.size());
    result.setConstant(false);

    for (Eigen::Index i = 1; i < x.size(); ++i) {
        if (i < x.size() - 1) {
            result(i) = (x(i) > x(i - 1)) && (x(i) >= x(i + 1));
        } else {
            result(i) = x(i) > x(i - 1);
        }
    }

    return result;
}

Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
localmax(const ArrayXXr& x, int axis) {
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(x.rows(), x.cols());
    result.setConstant(false);

    if (axis == 0) {
        // Along rows (comparing consecutive rows)
        for (Eigen::Index j = 0; j < x.cols(); ++j) {
            for (Eigen::Index i = 1; i < x.rows(); ++i) {
                if (i < x.rows() - 1) {
                    result(i, j) = (x(i, j) > x(i - 1, j)) && (x(i, j) >= x(i + 1, j));
                } else {
                    result(i, j) = x(i, j) > x(i - 1, j);
                }
            }
        }
    } else {
        // Along columns (comparing consecutive columns)
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            for (Eigen::Index j = 1; j < x.cols(); ++j) {
                if (j < x.cols() - 1) {
                    result(i, j) = (x(i, j) > x(i, j - 1)) && (x(i, j) >= x(i, j + 1));
                } else {
                    result(i, j) = x(i, j) > x(i, j - 1);
                }
            }
        }
    }

    return result;
}

Eigen::Array<bool, Eigen::Dynamic, 1> localmin(const ArrayXr& x) {
    Eigen::Array<bool, Eigen::Dynamic, 1> result(x.size());
    result.setConstant(false);

    for (Eigen::Index i = 1; i < x.size(); ++i) {
        if (i < x.size() - 1) {
            result(i) = (x(i) < x(i - 1)) && (x(i) <= x(i + 1));
        } else {
            result(i) = x(i) < x(i - 1);
        }
    }

    return result;
}

Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
localmin(const ArrayXXr& x, int axis) {
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(x.rows(), x.cols());
    result.setConstant(false);

    if (axis == 0) {
        for (Eigen::Index j = 0; j < x.cols(); ++j) {
            for (Eigen::Index i = 1; i < x.rows(); ++i) {
                if (i < x.rows() - 1) {
                    result(i, j) = (x(i, j) < x(i - 1, j)) && (x(i, j) <= x(i + 1, j));
                } else {
                    result(i, j) = x(i, j) < x(i - 1, j);
                }
            }
        }
    } else {
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            for (Eigen::Index j = 1; j < x.cols(); ++j) {
                if (j < x.cols() - 1) {
                    result(i, j) = (x(i, j) < x(i, j - 1)) && (x(i, j) <= x(i, j + 1));
                } else {
                    result(i, j) = x(i, j) < x(i, j - 1);
                }
            }
        }
    }

    return result;
}

std::vector<Eigen::Index> peak_pick(const ArrayXr& x,
                                    int pre_max, int post_max,
                                    int pre_avg, int post_avg,
                                    Real delta, int wait) {
    if (pre_max < 0 || post_max < 1 || pre_avg < 0 || post_avg < 1 ||
        delta < 0 || wait < 0) {
        throw ParameterError("Invalid peak_pick parameters");
    }

    std::vector<Eigen::Index> peaks;
    Eigen::Index n = x.size();

    // Special case first frame
    Real max_val = x.head(std::min(static_cast<Eigen::Index>(post_max), n)).maxCoeff();
    Real mean_val = x.head(std::min(static_cast<Eigen::Index>(post_avg), n)).mean();

    Eigen::Index i = 0;
    if (x(0) >= max_val && x(0) >= mean_val + delta) {
        peaks.push_back(0);
        i = wait + 1;
    } else {
        i = 1;
    }

    while (i < n) {
        Eigen::Index start_max = std::max(Eigen::Index(0), i - pre_max);
        Eigen::Index end_max = std::min(n, i + post_max);
        max_val = x.segment(start_max, end_max - start_max).maxCoeff();

        if (x(i) != max_val) {
            ++i;
            continue;
        }

        Eigen::Index start_avg = std::max(Eigen::Index(0), i - pre_avg);
        Eigen::Index end_avg = std::min(n, i + post_avg);
        mean_val = x.segment(start_avg, end_avg - start_avg).mean();

        if (x(i) >= mean_val + delta) {
            peaks.push_back(i);
            i += wait + 1;
        } else {
            ++i;
        }
    }

    return peaks;
}

ArrayXXr axis_sort(const ArrayXXr& S, int axis) {
    if (axis < 0) {
        axis = 2 + axis;
    }

    ArrayXXr result = S;

    if (axis == 0) {
        // Sort rows by argmax along columns
        std::vector<std::pair<Eigen::Index, Eigen::Index>> indices;
        for (Eigen::Index i = 0; i < S.rows(); ++i) {
            Eigen::Index max_idx;
            S.row(i).maxCoeff(&max_idx);
            indices.emplace_back(max_idx, i);
        }
        std::sort(indices.begin(), indices.end());

        for (Eigen::Index i = 0; i < S.rows(); ++i) {
            result.row(i) = S.row(indices[i].second);
        }
    } else {
        // Sort columns by argmax along rows
        std::vector<std::pair<Eigen::Index, Eigen::Index>> indices;
        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            Eigen::Index max_idx;
            S.col(j).maxCoeff(&max_idx);
            indices.emplace_back(max_idx, j);
        }
        std::sort(indices.begin(), indices.end());

        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            result.col(j) = S.col(indices[j].second);
        }
    }

    return result;
}

std::pair<ArrayXXr, std::vector<Eigen::Index>>
axis_sort_with_index(const ArrayXXr& S, int axis) {
    if (axis < 0) {
        axis = 2 + axis;
    }

    ArrayXXr result = S;
    std::vector<Eigen::Index> idx;

    if (axis == 0) {
        std::vector<std::pair<Eigen::Index, Eigen::Index>> indices;
        for (Eigen::Index i = 0; i < S.rows(); ++i) {
            Eigen::Index max_idx;
            S.row(i).maxCoeff(&max_idx);
            indices.emplace_back(max_idx, i);
        }
        std::sort(indices.begin(), indices.end());

        for (Eigen::Index i = 0; i < S.rows(); ++i) {
            result.row(i) = S.row(indices[i].second);
            idx.push_back(indices[i].second);
        }
    } else {
        std::vector<std::pair<Eigen::Index, Eigen::Index>> indices;
        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            Eigen::Index max_idx;
            S.col(j).maxCoeff(&max_idx);
            indices.emplace_back(max_idx, j);
        }
        std::sort(indices.begin(), indices.end());

        for (Eigen::Index j = 0; j < S.cols(); ++j) {
            result.col(j) = S.col(indices[j].second);
            idx.push_back(indices[j].second);
        }
    }

    return {result, idx};
}

ArrayXXr sparsify_rows(const ArrayXXr& x, Real quantile) {
    if (quantile < 0 || quantile >= 1) {
        throw ParameterError("quantile must be in [0, 1)");
    }

    ArrayXXr result = x;

    for (Eigen::Index i = 0; i < x.rows(); ++i) {
        ArrayXr row = x.row(i);
        std::vector<Real> sorted(row.data(), row.data() + row.size());
        std::sort(sorted.begin(), sorted.end());

        size_t idx = static_cast<size_t>(quantile * sorted.size());
        Real threshold = sorted[idx];

        for (Eigen::Index j = 0; j < x.cols(); ++j) {
            if (result(i, j) < threshold) {
                result(i, j) = 0;
            }
        }
    }

    return result;
}

ArrayXXr softmask(const ArrayXXr& X, const ArrayXXr& X_ref,
                  Real power, bool split_zeros) {
    if (X.rows() != X_ref.rows() || X.cols() != X_ref.cols()) {
        throw ParameterError("X and X_ref must have the same shape");
    }
    if ((X < 0).any() || (X_ref < 0).any()) {
        throw ParameterError("X and X_ref must be non-negative");
    }
    if (power <= 0) {
        throw ParameterError("power must be strictly positive");
    }

    ArrayXXr X_pow = X.abs().pow(power);
    ArrayXXr ref_pow = X_ref.abs().pow(power);
    ArrayXXr total = X_pow + ref_pow;

    ArrayXXr mask = ArrayXXr::Zero(X.rows(), X.cols());

    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            if (total(i, j) > 0) {
                mask(i, j) = X_pow(i, j) / total(i, j);
            } else if (split_zeros) {
                mask(i, j) = 0.5;
            }
        }
    }

    return mask;
}

ArrayXXr sync(const ArrayXXr& data, const std::vector<Eigen::Index>& idx,
              AggregateFunc aggregate, bool pad, int axis) {
    if (axis < 0) {
        axis = 2 + axis;
    }

    if (idx.empty()) {
        throw ParameterError("idx cannot be empty");
    }

    // Create segments
    std::vector<std::pair<Eigen::Index, Eigen::Index>> segments;
    for (size_t i = 0; i < idx.size(); ++i) {
        Eigen::Index start = idx[i];
        Eigen::Index end = (i + 1 < idx.size()) ? idx[i + 1] :
                          (axis == 0 ? data.rows() : data.cols());
        if (start < end) {
            segments.emplace_back(start, end);
        }
    }

    Eigen::Index n_segments = segments.size();
    ArrayXXr result;

    if (axis == 0) {
        result.resize(n_segments, data.cols());
        for (size_t s = 0; s < segments.size(); ++s) {
            auto [start, end] = segments[s];
            ArrayXXr segment = data.block(start, 0, end - start, data.cols());

            switch (aggregate) {
                case AggregateFunc::Mean:
                    result.row(s) = segment.colwise().mean();
                    break;
                case AggregateFunc::Median: {
                    // Compute median for each column
                    for (Eigen::Index j = 0; j < data.cols(); ++j) {
                        // Copy column to contiguous storage (ArrayXXr is RowMajor)
                        ArrayXr col_data = segment.col(j);
                        std::vector<Real> col(col_data.data(),
                                              col_data.data() + col_data.size());
                        std::sort(col.begin(), col.end());
                        result(s, j) = col[col.size() / 2];
                    }
                    break;
                }
                case AggregateFunc::Min:
                    result.row(s) = segment.colwise().minCoeff();
                    break;
                case AggregateFunc::Max:
                    result.row(s) = segment.colwise().maxCoeff();
                    break;
            }
        }
    } else {
        result.resize(data.rows(), n_segments);
        for (size_t s = 0; s < segments.size(); ++s) {
            auto [start, end] = segments[s];
            ArrayXXr segment = data.block(0, start, data.rows(), end - start);

            switch (aggregate) {
                case AggregateFunc::Mean:
                    result.col(s) = segment.rowwise().mean();
                    break;
                case AggregateFunc::Median: {
                    for (Eigen::Index i = 0; i < data.rows(); ++i) {
                        std::vector<Real> row(segment.row(i).data(),
                                              segment.row(i).data() + segment.cols());
                        std::sort(row.begin(), row.end());
                        result(i, s) = row[row.size() / 2];
                    }
                    break;
                }
                case AggregateFunc::Min:
                    result.col(s) = segment.rowwise().minCoeff();
                    break;
                case AggregateFunc::Max:
                    result.col(s) = segment.rowwise().maxCoeff();
                    break;
            }
        }
    }

    return result;
}

ArrayXr abs2(const ArrayXc& x) {
    return (x.real().square() + x.imag().square());
}

ArrayXXr abs2(const ArrayXXc& x) {
    return (x.real().square() + x.imag().square());
}

ArrayXc phasor(const ArrayXr& angles, const ArrayXr& mag) {
    if (angles.size() != mag.size()) {
        throw ParameterError("angles and mag must have same size");
    }
    ArrayXc result(angles.size());
    for (Eigen::Index i = 0; i < angles.size(); ++i) {
        result(i) = std::polar(mag(i), angles(i));
    }
    return result;
}

ArrayXc phasor(const ArrayXr& angles, Real mag) {
    ArrayXc result(angles.size());
    for (Eigen::Index i = 0; i < angles.size(); ++i) {
        result(i) = std::polar(mag, angles(i));
    }
    return result;
}

ArrayXr cyclic_gradient(const ArrayXr& x, int edge_order) {
    Eigen::Index n = x.size();

    // Wrap-pad the data by edge_order on each side
    Eigen::Index pad_len = n + 2 * edge_order;
    ArrayXr x_pad(pad_len);

    // Left wrap padding: last edge_order elements
    for (int i = 0; i < edge_order; ++i) {
        x_pad(i) = x(n - edge_order + i);
    }
    // Original data
    x_pad.segment(edge_order, n) = x;
    // Right wrap padding: first edge_order elements
    for (int i = 0; i < edge_order; ++i) {
        x_pad(edge_order + n + i) = x(i);
    }

    // Compute np.gradient equivalent on padded data
    ArrayXr grad_pad(pad_len);

    // Central differences for interior (all points except first and last)
    for (Eigen::Index i = 1; i < pad_len - 1; ++i) {
        grad_pad(i) = (x_pad(i + 1) - x_pad(i - 1)) / 2.0;
    }

    // Edge handling for padded array
    if (edge_order == 1) {
        grad_pad(0) = x_pad(1) - x_pad(0);
        grad_pad(pad_len - 1) = x_pad(pad_len - 1) - x_pad(pad_len - 2);
    } else {
        // Second-order forward/backward differences at edges
        grad_pad(0) = (-3.0 * x_pad(0) + 4.0 * x_pad(1) - x_pad(2)) / 2.0;
        grad_pad(pad_len - 1) = (3.0 * x_pad(pad_len - 1) - 4.0 * x_pad(pad_len - 2) + x_pad(pad_len - 3)) / 2.0;
    }

    // Strip padding
    ArrayXr grad = grad_pad.segment(edge_order, n);

    return grad;
}

ArrayXXr fill_off_diagonal(const ArrayXXr& x, Real fill_value, bool wrap) {
    if (x.rows() != x.cols()) {
        throw ParameterError("Input must be square");
    }

    ArrayXXr result = x;
    Eigen::Index n = x.rows();

    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            if (i != j) {
                result(i, j) = fill_value;
            }
        }
    }

    return result;
}

ArrayXXr stack(const std::vector<ArrayXr>& arrays, int axis) {
    if (arrays.empty()) {
        throw ParameterError("Cannot stack empty list");
    }

    Eigen::Index max_len = arrays[0].size();
    for (size_t i = 1; i < arrays.size(); ++i) {
        if (arrays[i].size() != max_len) {
            throw ParameterError("All arrays must have the same shape for stacking");
        }
    }

    if (axis == 0) {
        ArrayXXr result(arrays.size(), max_len);
        result.setZero();
        for (size_t i = 0; i < arrays.size(); ++i) {
            result.row(i).head(arrays[i].size()) = arrays[i].transpose();
        }
        return result;
    } else {
        ArrayXXr result(max_len, arrays.size());
        result.setZero();
        for (size_t j = 0; j < arrays.size(); ++j) {
            result.col(j).head(arrays[j].size()) = arrays[j];
        }
        return result;
    }
}

ArrayXr buf_to_float(const void* buf, size_t n_samples, int n_bytes) {
    ArrayXr result(n_samples);

    if (n_bytes == 2) {
        // int16
        const int16_t* ptr = static_cast<const int16_t*>(buf);
        for (size_t i = 0; i < n_samples; ++i) {
            result(i) = static_cast<Real>(ptr[i]) / 32768.0;
        }
    } else if (n_bytes == 4) {
        // int32 or float
        const int32_t* ptr = static_cast<const int32_t*>(buf);
        for (size_t i = 0; i < n_samples; ++i) {
            result(i) = static_cast<Real>(ptr[i]) / 2147483648.0;
        }
    } else if (n_bytes == 1) {
        // uint8
        const uint8_t* ptr = static_cast<const uint8_t*>(buf);
        for (size_t i = 0; i < n_samples; ++i) {
            result(i) = (static_cast<Real>(ptr[i]) - 128.0) / 128.0;
        }
    } else {
        throw ParameterError("Unsupported byte format");
    }

    return result;
}

bool is_unique(const ArrayXr& x) {
    std::set<Real> seen;
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        if (!seen.insert(x(i)).second) {
            return false;
        }
    }
    return true;
}

size_t count_unique(const ArrayXr& x) {
    std::set<Real> seen(x.data(), x.data() + x.size());
    return seen.size();
}

ArrayXXr shear(const ArrayXXr& X, int factor, int axis) {
    if (axis < 0) {
        axis = 2 + axis;
    }

    ArrayXXr result = X;

    if (axis == 0) {
        // Shear along rows
        for (Eigen::Index i = 0; i < X.rows(); ++i) {
            Eigen::Index shift = (i * factor) % X.cols();
            if (shift < 0) shift += X.cols();
            // Circular shift the row
            ArrayXr row = X.row(i);
            for (Eigen::Index j = 0; j < X.cols(); ++j) {
                result(i, j) = row((j - shift + X.cols()) % X.cols());
            }
        }
    } else {
        // Shear along columns
        for (Eigen::Index j = 0; j < X.cols(); ++j) {
            Eigen::Index shift = (j * factor) % X.rows();
            if (shift < 0) shift += X.rows();
            ArrayXr col = X.col(j);
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                result(i, j) = col((i - shift + X.rows()) % X.rows());
            }
        }
    }

    return result;
}

std::vector<Eigen::Index> index_to_slice(Eigen::Index idx,
                                          Eigen::Index idx_min,
                                          std::optional<Eigen::Index> idx_max,
                                          Eigen::Index step,
                                          bool pad) {
    std::vector<Eigen::Index> result;

    Eigen::Index start = pad ? idx_min : idx;
    Eigen::Index end = idx_max.value_or(idx + 1);

    for (Eigen::Index i = start; i < end; i += step) {
        result.push_back(i);
    }

    return result;
}

// ============================================================================
// Match Intervals (Jaccard similarity)
// ============================================================================

namespace {

Real jaccard(Real a0, Real a1, Real b0, Real b1) {
    Real min_end = std::min(a1, b1);
    Real max_end = std::max(a1, b1);
    Real min_start = std::min(a0, b0);
    Real max_start = std::max(a0, b0);

    Real intersection = min_end - max_start;
    if (intersection < 0) intersection = 0.0;

    Real union_val = max_end - min_start;
    if (union_val > 0) return intersection / union_val;
    return 0.0;
}

} // anonymous namespace

std::vector<Eigen::Index> match_intervals(
    const ArrayXXr& intervals_from,
    const ArrayXXr& intervals_to,
    bool strict) {

    if (intervals_from.rows() == 0 || intervals_to.rows() == 0) {
        throw ParameterError("Attempting to match empty interval list");
    }

    valid_intervals(intervals_from);
    valid_intervals(intervals_to);

    Eigen::Index n = intervals_from.rows();
    Eigen::Index m = intervals_to.rows();

    // Build sort indices by start and end times
    std::vector<Eigen::Index> start_index(m), end_index(m);
    std::iota(start_index.begin(), start_index.end(), 0);
    std::iota(end_index.begin(), end_index.end(), 0);

    std::sort(start_index.begin(), start_index.end(),
        [&](Eigen::Index a, Eigen::Index b) {
            return intervals_to(a, 0) < intervals_to(b, 0);
        });
    std::sort(end_index.begin(), end_index.end(),
        [&](Eigen::Index a, Eigen::Index b) {
            return intervals_to(a, 1) < intervals_to(b, 1);
        });

    // Sorted values of starts and ends
    std::vector<Real> start_sorted(m), end_sorted(m);
    for (Eigen::Index i = 0; i < m; ++i) {
        start_sorted[i] = intervals_to(start_index[i], 0);
        end_sorted[i] = intervals_to(end_index[i], 1);
    }

    std::vector<Eigen::Index> output(n);

    for (Eigen::Index i = 0; i < n; ++i) {
        Real query_start = intervals_from(i, 0);
        Real query_end = intervals_from(i, 1);

        // searchsorted(start_sorted, query_end, side='right')
        Eigen::Index after_query = static_cast<Eigen::Index>(
            std::upper_bound(start_sorted.begin(), start_sorted.end(), query_end)
            - start_sorted.begin());

        // searchsorted(end_sorted, query_start, side='left')
        Eigen::Index before_query = static_cast<Eigen::Index>(
            std::lower_bound(end_sorted.begin(), end_sorted.end(), query_start)
            - end_sorted.begin());

        // Candidates: intervals that start before query ends AND end after query starts
        std::set<Eigen::Index> set_start(start_index.begin(), start_index.begin() + after_query);
        std::set<Eigen::Index> set_end(end_index.begin() + before_query, end_index.end());

        // Intersection of the two sets
        std::vector<Eigen::Index> candidates;
        std::set_intersection(set_start.begin(), set_start.end(),
                              set_end.begin(), set_end.end(),
                              std::back_inserter(candidates));

        if (!candidates.empty()) {
            // Find best Jaccard match
            Real best_score = -1;
            Eigen::Index best_idx = -1;
            for (auto idx : candidates) {
                Real score = jaccard(query_start, query_end,
                                     intervals_to(idx, 0), intervals_to(idx, 1));
                if (score > best_score) {
                    best_score = score;
                    best_idx = idx;
                }
            }
            output[i] = best_idx;
        } else if (strict) {
            throw ParameterError("Unable to match intervals with strict=true");
        } else {
            // Find nearest disjoint interval
            Real dist_before = std::numeric_limits<Real>::infinity();
            Real dist_after = std::numeric_limits<Real>::infinity();

            if (before_query > 0) {
                dist_before = query_start - end_sorted[before_query - 1];
            }
            if (after_query < m) {
                dist_after = start_sorted[after_query] - query_end;
            }

            if (dist_before <= dist_after) {
                output[i] = end_index[before_query - 1];
            } else {
                output[i] = start_index[after_query];
            }
        }
    }

    return output;
}

// ============================================================================
// Non-negative Least Squares (NNLS)
// ============================================================================

} // namespace util
} // namespace librosa

// Include fnnls after closing namespace to avoid conflicts
#include "fnnls/fnnls.hpp"

namespace librosa {
namespace util {

ArrayXXr nnls(const MatrixXr& A, const ArrayXXr& B) {
    // A: (m, n), B: (m, k) — solve for X: (n, k) where X >= 0
    // fnnls expects ZT as a ColMajor matrix of shape (n, m)
    // ZT = A^T

    Eigen::Index m = A.rows();
    Eigen::Index n = A.cols();
    Eigen::Index k = B.cols();

    if (B.rows() != m) {
        throw ParameterError("A and B have incompatible dimensions for NNLS");
    }

    // Compute ZT = A^T as ColMajor (n, m) for fnnls compatibility
    using ColMajorMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
    ColMajorMatrix ZT = ColMajorMatrix(A.transpose());

    // Precompute ZTZ = ZT * ZT^T = A^T * A, shape (n, n), also ColMajor
    Eigen::SparseMatrix<Real> ZT_sp = ZT.sparseView();
    ColMajorMatrix ZTZ = ColMajorMatrix::Zero(n, n);
    ZTZ.template triangularView<Eigen::Upper>() = ColMajorMatrix(ZT_sp * ZT_sp.transpose());
    ZTZ.template triangularView<Eigen::Lower>() = ZTZ.transpose();

    ArrayXXr X(n, k);

    // Solve column-by-column
    for (Eigen::Index col = 0; col < k; ++col) {
        Eigen::Map<fnnls::MatrixX_<Real>> ZT_map(ZT.data(), n, m);

        // Copy column of B since B is RowMajor (columns not contiguous)
        Eigen::Matrix<Real, Eigen::Dynamic, 1> b_col = B.col(col).matrix();
        Eigen::Map<fnnls::VectorX_<Real>> x_map(b_col.data(), m);

        try {
            fnnls::VectorX_<Real> result = fnnls::fnnls_solver<Real>(
                ZT_map, x_map, 3 * static_cast<int>(n), -1.0, ZTZ.data());
            X.col(col) = result.array();
        } catch (const std::runtime_error&) {
            // If solver doesn't converge, use zero column
            X.col(col).setZero();
        }
    }

    return X;
}

} // namespace util
} // namespace librosa
