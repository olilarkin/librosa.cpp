#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace librosa {

// Scalar types
using Real = double;
using Complex = std::complex<Real>;

// Array types (following NumPy conventions)
using ArrayXr = Eigen::Array<Real, Eigen::Dynamic, 1>;
using ArrayXc = Eigen::Array<Complex, Eigen::Dynamic, 1>;
using ArrayXXr = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ArrayXXc = Eigen::Array<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Matrix types
using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using VectorXc = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using MatrixXr = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SparseMatrixXr = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using SparseMatrixXc = Eigen::SparseMatrix<Complex, Eigen::RowMajor>;

// Map types for zero-copy operations
using MapVectorXr = Eigen::Map<VectorXr>;
using MapMatrixXr = Eigen::Map<MatrixXr>;
using MapArrayXr = Eigen::Map<ArrayXr>;
using MapArrayXXr = Eigen::Map<ArrayXXr>;

using ConstMapVectorXr = Eigen::Map<const VectorXr>;
using ConstMapMatrixXr = Eigen::Map<const MatrixXr>;
using ConstMapArrayXr = Eigen::Map<const ArrayXr>;
using ConstMapArrayXXr = Eigen::Map<const ArrayXXr>;

// Audio-specific types
struct AudioData {
    ArrayXXr samples;  // (channels, samples) or (1, samples) for mono
    Real sample_rate;
    int channels;

    // Convenience accessors
    Eigen::Index num_samples() const { return samples.cols(); }
    Eigen::Index num_channels() const { return samples.rows(); }
    Real duration() const { return static_cast<Real>(num_samples()) / sample_rate; }

    // Get mono representation (averages channels if stereo)
    ArrayXr mono() const;
};

// Padding mode enum (mirrors numpy.pad modes)
enum class PadMode {
    Constant,   // Pad with constant value (default 0)
    Edge,       // Pad with edge values
    Reflect,    // Reflect signal (d c b a | a b c d | d c b a)
    Symmetric,  // Symmetric padding (d c b | a b c d | c b a)
    Wrap,       // Wrap around
    Linear_ramp // Linear ramp to end_value
};

// Window types
enum class WindowType {
    Hann,
    Hamming,
    Blackman,
    Bartlett,
    Rectangular,
    Kaiser,
    Gaussian,
    Tukey,
    Triangle
};

// Normalization modes
enum class NormMode {
    None,
    Slaney,     // Slaney-style mel normalization
    L1,
    L2,
    Max
};

// Interpolation types
enum class InterpType {
    Linear,
    Nearest,
    Cubic
};

// Frequency scale types
enum class FreqScale {
    Linear,
    Log,
    Mel,
    CQT
};

// HTK vs Slaney mel scale
enum class MelScale {
    HTK,
    Slaney
};

// Aggregation functions for sync
enum class AggregateFunc {
    Mean,
    Median,
    Min,
    Max
};

// Constants
namespace constants {
    constexpr Real PI = 3.14159265358979323846;
    constexpr Real TWO_PI = 2.0 * PI;
    constexpr Real A4_FREQ = 440.0;        // Reference frequency for A4
    constexpr int A4_MIDI = 69;            // MIDI note number for A4
    constexpr Real TINY = 1e-10;           // Small value to avoid log(0)
    constexpr int DEFAULT_SR = 22050;      // Default sample rate
    constexpr int DEFAULT_N_FFT = 2048;    // Default FFT size
    constexpr int DEFAULT_HOP_LENGTH = 512;// Default hop length
    constexpr int DEFAULT_N_MELS = 128;    // Default number of mel bands
    constexpr int DEFAULT_N_MFCC = 20;     // Default number of MFCCs
}

// Result type for functions that may fail
template<typename T>
using Result = std::variant<T, std::string>;

} // namespace librosa
