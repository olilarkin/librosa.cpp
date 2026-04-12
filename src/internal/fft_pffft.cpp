#include "fft.hpp"

#include <librosa/util/exceptions.hpp>

#include <pffft/pffft_double.h>

#include <string>
#include <vector>
#include <new>

// pffftd only accepts lengths factorable as 2^a * 3^b * 5^c (and >= 32 for
// real transforms, 16 for complex). Sizes outside that family are reported as
// an exception here — the pffft backend is targeted at environments where
// FFTW isn't available (Emscripten/WASM) and tests on that path are
// build-only for now.

namespace librosa {
namespace internal {

namespace {
    [[noreturn]] void unsupported_length(int n_fft, const char* which) {
        throw ParameterError("pffft FFT backend does not support " +
                             std::string(which) + " length " + std::to_string(n_fft));
    }

    struct AlignedDoubles {
        double* ptr = nullptr;
        size_t size = 0;

        explicit AlignedDoubles(size_t n) : ptr(static_cast<double*>(pffftd_aligned_malloc(n * sizeof(double)))), size(n) {
            if (!ptr) {
                throw std::bad_alloc();
            }
        }

        ~AlignedDoubles() {
            if (ptr) pffftd_aligned_free(ptr);
        }

        AlignedDoubles(const AlignedDoubles&) = delete;
        AlignedDoubles& operator=(const AlignedDoubles&) = delete;

        double* data() { return ptr; }
        const double* data() const { return ptr; }
    };
}

struct RealFft::Impl {
    int n_fft;
    PFFFTD_Setup* setup = nullptr;
    AlignedDoubles work;       // 2*N scratch for pffftd
    AlignedDoubles scratch_in; // N real samples
    AlignedDoubles scratch_out;// N doubles: packed interleaved complex

    explicit Impl(int n)
        : n_fft(n),
          work(static_cast<size_t>(2 * n)),
          scratch_in(static_cast<size_t>(n)),
          scratch_out(static_cast<size_t>(n)) {
        if (n <= 0 || !pffftd_is_valid_size(n, PFFFT_REAL)) {
            unsupported_length(n, "real FFT");
        }
        setup = pffftd_new_setup(n, PFFFT_REAL);
        if (!setup) unsupported_length(n, "real FFT");
    }

    ~Impl() {
        if (setup) pffftd_destroy_setup(setup);
    }
};

RealFft::RealFft(int n_fft)
    : impl_(std::make_unique<Impl>(n_fft)), n_fft_(n_fft) {}

RealFft::~RealFft() = default;
RealFft::RealFft(RealFft&&) noexcept = default;
RealFft& RealFft::operator=(RealFft&&) noexcept = default;

void RealFft::forward(const Real* in, Complex* out) {
    const int N = impl_->n_fft;
    const int H = N / 2;

    std::copy(in, in + N, impl_->scratch_in.data());
    pffftd_transform_ordered(impl_->setup,
                             impl_->scratch_in.data(),
                             impl_->scratch_out.data(),
                             impl_->work.data(),
                             PFFFT_FORWARD);

    // pffft_ordered real output: scratch_out[0] = DC, scratch_out[1] = Nyquist,
    // scratch_out[2k], scratch_out[2k+1] = bin k complex for 1 <= k < N/2.
    const double* s = impl_->scratch_out.data();
    out[0] = Complex(s[0], 0.0);
    out[H] = Complex(s[1], 0.0);
    for (int k = 1; k < H; ++k) {
        out[k] = Complex(s[2 * k], s[2 * k + 1]);
    }
}

void RealFft::inverse(const Complex* in, Real* out) {
    const int N = impl_->n_fft;
    const int H = N / 2;

    double* s = impl_->scratch_out.data();
    s[0] = in[0].real();
    s[1] = in[H].real();
    for (int k = 1; k < H; ++k) {
        s[2 * k]     = in[k].real();
        s[2 * k + 1] = in[k].imag();
    }

    pffftd_transform_ordered(impl_->setup,
                             s,
                             impl_->scratch_in.data(),
                             impl_->work.data(),
                             PFFFT_BACKWARD);

    // pffftd BACKWARD output is already unscaled (N * h[j]) — matches FFTW.
    std::copy(impl_->scratch_in.data(), impl_->scratch_in.data() + N, out);
}

struct ComplexFft::Impl {
    int n_fft;
    PFFFTD_Setup* setup = nullptr;
    AlignedDoubles work;
    AlignedDoubles scratch_in;   // 2*N doubles (interleaved complex)
    AlignedDoubles scratch_out;

    explicit Impl(int n)
        : n_fft(n),
          work(static_cast<size_t>(2 * n)),
          scratch_in(static_cast<size_t>(2 * n)),
          scratch_out(static_cast<size_t>(2 * n)) {
        if (n <= 0 || !pffftd_is_valid_size(n, PFFFT_COMPLEX)) {
            unsupported_length(n, "complex FFT");
        }
        setup = pffftd_new_setup(n, PFFFT_COMPLEX);
        if (!setup) unsupported_length(n, "complex FFT");
    }

    ~Impl() {
        if (setup) pffftd_destroy_setup(setup);
    }
};

ComplexFft::ComplexFft(int n_fft)
    : impl_(std::make_unique<Impl>(n_fft)), n_fft_(n_fft) {}

ComplexFft::~ComplexFft() = default;
ComplexFft::ComplexFft(ComplexFft&&) noexcept = default;
ComplexFft& ComplexFft::operator=(ComplexFft&&) noexcept = default;

void ComplexFft::forward(const Complex* in, Complex* out) {
    const int N = impl_->n_fft;
    double* s = impl_->scratch_in.data();
    for (int i = 0; i < N; ++i) {
        s[2 * i]     = in[i].real();
        s[2 * i + 1] = in[i].imag();
    }
    pffftd_transform_ordered(impl_->setup,
                             s,
                             impl_->scratch_out.data(),
                             impl_->work.data(),
                             PFFFT_FORWARD);
    const double* o = impl_->scratch_out.data();
    for (int i = 0; i < N; ++i) {
        out[i] = Complex(o[2 * i], o[2 * i + 1]);
    }
}

} // namespace internal
} // namespace librosa
