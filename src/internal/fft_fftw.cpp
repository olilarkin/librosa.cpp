#include "fft.hpp"

#include <fftw3.h>

#include <cstring>

namespace librosa {
namespace internal {

namespace {
    // std::complex<double> has the same object representation as fftw_complex
    // (two consecutive doubles). Reinterpret freely on both sides.
    fftw_complex* as_fftw(Complex* p) {
        return reinterpret_cast<fftw_complex*>(p);
    }
    const fftw_complex* as_fftw(const Complex* p) {
        return reinterpret_cast<const fftw_complex*>(p);
    }
}

struct RealFft::Impl {
    int n_fft;
    fftw_plan forward_plan = nullptr;
    fftw_plan inverse_plan = nullptr;
    double* scratch_real = nullptr;
    fftw_complex* scratch_complex = nullptr;

    explicit Impl(int n) : n_fft(n) {
        scratch_real = fftw_alloc_real(static_cast<size_t>(n));
        scratch_complex = fftw_alloc_complex(static_cast<size_t>(n / 2 + 1));
        // FFTW_UNALIGNED lets fftw_execute_dft_{r2c,c2r} accept any caller buffer.
        const unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
        forward_plan = fftw_plan_dft_r2c_1d(n_fft, scratch_real,
                                            scratch_complex, flags);
        inverse_plan = fftw_plan_dft_c2r_1d(n_fft, scratch_complex,
                                            scratch_real, flags);
    }

    ~Impl() {
        if (forward_plan) fftw_destroy_plan(forward_plan);
        if (inverse_plan) fftw_destroy_plan(inverse_plan);
        if (scratch_real) fftw_free(scratch_real);
        if (scratch_complex) fftw_free(scratch_complex);
    }
};

RealFft::RealFft(int n_fft)
    : impl_(std::make_unique<Impl>(n_fft)), n_fft_(n_fft) {}

RealFft::~RealFft() = default;
RealFft::RealFft(RealFft&&) noexcept = default;
RealFft& RealFft::operator=(RealFft&&) noexcept = default;

void RealFft::forward(const Real* in, Complex* out) {
    fftw_execute_dft_r2c(impl_->forward_plan,
                         const_cast<double*>(in),
                         as_fftw(out));
}

void RealFft::inverse(const Complex* in, Real* out) {
    fftw_execute_dft_c2r(impl_->inverse_plan,
                         const_cast<fftw_complex*>(as_fftw(in)),
                         out);
}

struct ComplexFft::Impl {
    int n_fft;
    fftw_plan forward_plan = nullptr;
    fftw_complex* scratch_in = nullptr;
    fftw_complex* scratch_out = nullptr;

    explicit Impl(int n) : n_fft(n) {
        scratch_in = fftw_alloc_complex(static_cast<size_t>(n));
        scratch_out = fftw_alloc_complex(static_cast<size_t>(n));
        const unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
        forward_plan = fftw_plan_dft_1d(n_fft, scratch_in, scratch_out,
                                        FFTW_FORWARD, flags);
    }

    ~Impl() {
        if (forward_plan) fftw_destroy_plan(forward_plan);
        if (scratch_in) fftw_free(scratch_in);
        if (scratch_out) fftw_free(scratch_out);
    }
};

ComplexFft::ComplexFft(int n_fft)
    : impl_(std::make_unique<Impl>(n_fft)), n_fft_(n_fft) {}

ComplexFft::~ComplexFft() = default;
ComplexFft::ComplexFft(ComplexFft&&) noexcept = default;
ComplexFft& ComplexFft::operator=(ComplexFft&&) noexcept = default;

void ComplexFft::forward(const Complex* in, Complex* out) {
    fftw_execute_dft(impl_->forward_plan,
                     const_cast<fftw_complex*>(as_fftw(in)),
                     as_fftw(out));
}

} // namespace internal
} // namespace librosa
