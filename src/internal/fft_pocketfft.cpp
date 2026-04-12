#include "fft.hpp"

#include <librosa/util/exceptions.hpp>

#include <pocketfft_hdronly.h>

#include <cstddef>
#include <vector>

// PocketFFT header-only backend. Supports arbitrary transform lengths,
// including primes > 5 via Bluestein internally. Matches FFTW's unscaled
// convention by passing fct=1.0 in both directions.

namespace librosa {
namespace internal {

namespace {
    pocketfft::shape_t make_shape(int n) {
        return pocketfft::shape_t{static_cast<std::size_t>(n)};
    }

    pocketfft::stride_t stride_real() {
        return pocketfft::stride_t{static_cast<std::ptrdiff_t>(sizeof(double))};
    }

    pocketfft::stride_t stride_complex() {
        return pocketfft::stride_t{
            static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
    }
}

struct RealFft::Impl {
    int n_fft;
    explicit Impl(int n) : n_fft(n) {
        if (n <= 0) {
            throw ParameterError("RealFft requires positive n_fft");
        }
    }
};

RealFft::RealFft(int n_fft)
    : impl_(std::make_unique<Impl>(n_fft)), n_fft_(n_fft) {}

RealFft::~RealFft() = default;
RealFft::RealFft(RealFft&&) noexcept = default;
RealFft& RealFft::operator=(RealFft&&) noexcept = default;

void RealFft::forward(const Real* in, Complex* out) {
    const int N = impl_->n_fft;
    pocketfft::r2c(make_shape(N), stride_real(), stride_complex(),
                   /*axis=*/0, /*forward=*/true,
                   in, out, /*fct=*/1.0);
}

void RealFft::inverse(const Complex* in, Real* out) {
    const int N = impl_->n_fft;
    pocketfft::c2r(make_shape(N), stride_complex(), stride_real(),
                   /*axis=*/0, /*forward=*/false,
                   in, out, /*fct=*/1.0);
}

struct ComplexFft::Impl {
    int n_fft;
    explicit Impl(int n) : n_fft(n) {
        if (n <= 0) {
            throw ParameterError("ComplexFft requires positive n_fft");
        }
    }
};

ComplexFft::ComplexFft(int n_fft)
    : impl_(std::make_unique<Impl>(n_fft)), n_fft_(n_fft) {}

ComplexFft::~ComplexFft() = default;
ComplexFft::ComplexFft(ComplexFft&&) noexcept = default;
ComplexFft& ComplexFft::operator=(ComplexFft&&) noexcept = default;

void ComplexFft::forward(const Complex* in, Complex* out) {
    const int N = impl_->n_fft;
    pocketfft::c2c(make_shape(N), stride_complex(), stride_complex(),
                   pocketfft::shape_t{0}, /*forward=*/true,
                   in, out, /*fct=*/1.0);
}

} // namespace internal
} // namespace librosa
