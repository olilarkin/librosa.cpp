#pragma once

#include <librosa/types.hpp>
#include <memory>

namespace librosa {
namespace internal {

// Backend-agnostic FFT primitives. All implementations produce output matching
// FFTW's unscaled convention: forward transforms are unnormalized, inverse
// transforms are unnormalized and the caller divides by n_fft when needed.
//
// Plan state lives inside each instance, so reuse a single object across
// frames rather than constructing one per call.

class RealFft {
public:
    explicit RealFft(int n_fft);
    ~RealFft();

    RealFft(const RealFft&) = delete;
    RealFft& operator=(const RealFft&) = delete;
    RealFft(RealFft&&) noexcept;
    RealFft& operator=(RealFft&&) noexcept;

    // in: real buffer of length n_fft; out: complex buffer of length n_fft/2 + 1.
    void forward(const Real* in, Complex* out);

    // in: complex buffer of length n_fft/2 + 1; out: real buffer of length n_fft.
    void inverse(const Complex* in, Real* out);

    int size() const noexcept { return n_fft_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int n_fft_;
};

class ComplexFft {
public:
    explicit ComplexFft(int n_fft);
    ~ComplexFft();

    ComplexFft(const ComplexFft&) = delete;
    ComplexFft& operator=(const ComplexFft&) = delete;
    ComplexFft(ComplexFft&&) noexcept;
    ComplexFft& operator=(ComplexFft&&) noexcept;

    // Forward complex->complex, unscaled. Both buffers length n_fft.
    void forward(const Complex* in, Complex* out);

    int size() const noexcept { return n_fft_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int n_fft_;
};

} // namespace internal
} // namespace librosa
